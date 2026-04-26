import base64
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path

import PIL.Image
import PIL.ImageEnhance
import PIL.ImageOps
import requests
from dotenv import load_dotenv

from smolagents import Tool, tool


load_dotenv(override=True)

DEFAULT_VISUAL_CAPTION_MODEL_ID = os.getenv(
    "SMOLAGENTS_VISUAL_CAPTION_MODEL_ID",
    "Salesforce/blip-image-captioning-base",
)
DEFAULT_VISUAL_VQA_MODEL_ID = os.getenv(
    "SMOLAGENTS_VISUAL_VQA_MODEL_ID",
    "dandelin/vilt-b32-finetuned-vqa",
)
DEFAULT_VISUAL_OCR_MODEL_ID = os.getenv(
    "SMOLAGENTS_VISUAL_OCR_MODEL_ID",
    "microsoft/trocr-small-printed",
)
visual_model_cache_local = threading.local()

OCR_QUERY_KEYWORDS = (
    "text",
    "word",
    "number",
    "digit",
    "written",
    "write",
    "read",
    "says",
    "title",
    "label",
    "caption",
    "transcribe",
    "anagram",
    "spreadsheet",
    "table",
    "fraction",
    "fractions",
    "slash",
    "equation",
    "equations",
    "expression",
    "expressions",
    "math",
    "problem",
    "problems",
    "sample problem",
    "sample problems",
    "answer",
    "answers",
    "list",
    "extract",
    "identify",
    "order",
    "comma-separated",
)
OCR_TRANSCRIPTION_KEYWORDS = (
    "transcribe",
    "read the text",
    "what is written",
    "what does it say",
    "what word",
    "what words",
    "what text",
)
OCR_LISTING_KEYWORDS = (
    "list",
    "extract",
    "identify",
    "provide",
    "show",
    "which fractions",
    "what fractions",
    "sample problem",
    "sample problems",
    "fractions",
    "fraction",
    "answers",
    "answer",
    "order",
    "appear",
    "comma-separated",
    "slash",
)
OCR_NUMERIC_KEYWORDS = (
    "what number",
    "which number",
    "how many",
    "digit",
    "score",
    "year",
    "price",
    "percent",
    "percentage",
)
APPLE_VISION_OCR_SCRIPT = """
import AppKit
import Vision
import Foundation

let args = CommandLine.arguments
guard args.count > 1 else {
    fputs("missing image path\\n", stderr)
    exit(1)
}
let imageURL = URL(fileURLWithPath: args[1])
guard let image = NSImage(contentsOf: imageURL) else {
    fputs("failed to load image\\n", stderr)
    exit(2)
}
guard let tiff = image.tiffRepresentation,
      let bitmap = NSBitmapImageRep(data: tiff),
      let cgImage = bitmap.cgImage else {
    fputs("failed to make cgImage\\n", stderr)
    exit(3)
}

let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.usesLanguageCorrection = false
if #available(macOS 13.0, *) {
    request.automaticallyDetectsLanguage = true
}

let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
do {
    try handler.perform([request])
    let observations = request.results ?? []
    for observation in observations {
        if let candidate = observation.topCandidates(1).first {
            print(candidate.string)
        }
    }
} catch {
    fputs("vision error: \\(error)\\n", stderr)
    exit(4)
}
"""


def get_visual_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_visual_model_cache() -> dict[str, tuple[object, object]]:
    if not hasattr(visual_model_cache_local, "models"):
        visual_model_cache_local.models = {}
    return visual_model_cache_local.models


def get_visual_runtime_cache() -> dict[str, str]:
    if not hasattr(visual_model_cache_local, "runtime"):
        visual_model_cache_local.runtime = {}
    return visual_model_cache_local.runtime


def get_caption_components():
    from transformers import BlipForConditionalGeneration, BlipProcessor

    cache = get_visual_model_cache()
    cache_key = f"caption::{DEFAULT_VISUAL_CAPTION_MODEL_ID}"
    if cache_key not in cache:
        processor = BlipProcessor.from_pretrained(DEFAULT_VISUAL_CAPTION_MODEL_ID)
        model = BlipForConditionalGeneration.from_pretrained(
            DEFAULT_VISUAL_CAPTION_MODEL_ID,
            use_safetensors=True,
        )
        model.to(get_visual_device())
        model.eval()
        cache[cache_key] = (processor, model)
    return cache[cache_key]


def get_vqa_components():
    from transformers import ViltForQuestionAnswering, ViltProcessor

    cache = get_visual_model_cache()
    cache_key = f"vqa::{DEFAULT_VISUAL_VQA_MODEL_ID}"
    if cache_key not in cache:
        processor = ViltProcessor.from_pretrained(DEFAULT_VISUAL_VQA_MODEL_ID)
        model = ViltForQuestionAnswering.from_pretrained(
            DEFAULT_VISUAL_VQA_MODEL_ID,
            use_safetensors=True,
        )
        model.to(get_visual_device())
        model.eval()
        cache[cache_key] = (processor, model)
    return cache[cache_key]


def get_ocr_components():
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    cache = get_visual_model_cache()
    cache_key = f"ocr::{DEFAULT_VISUAL_OCR_MODEL_ID}"
    if cache_key not in cache:
        processor = TrOCRProcessor.from_pretrained(DEFAULT_VISUAL_OCR_MODEL_ID)
        model = VisionEncoderDecoderModel.from_pretrained(
            DEFAULT_VISUAL_OCR_MODEL_ID,
            use_safetensors=True,
        )
        model.to(get_visual_device())
        model.eval()
        cache[cache_key] = (processor, model)
    return cache[cache_key]


def materialize_image_path(image_path: str) -> str:
    if not image_path.startswith("http"):
        return image_path

    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
    request_kwargs = {
        "headers": {"User-Agent": user_agent},
        "stream": True,
    }

    response = requests.get(image_path, **request_kwargs)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")

    extension = mimetypes.guess_extension(content_type)
    if extension is None:
        extension = ".download"

    fname = str(uuid.uuid4()) + extension
    os.makedirs("downloads", exist_ok=True)
    download_path = os.path.abspath(os.path.join("downloads", fname))

    with open(download_path, "wb") as fh:
        for chunk in response.iter_content(chunk_size=512):
            fh.write(chunk)

    return download_path


def ensure_apple_vision_ocr_binary() -> str | None:
    if os.sys.platform != "darwin":
        return None
    if shutil.which("swiftc") is None:
        return None

    runtime_cache = get_visual_runtime_cache()
    cache_key = "apple_vision_ocr_binary"
    if cache_key in runtime_cache and os.path.exists(runtime_cache[cache_key]):
        return runtime_cache[cache_key]

    cache_dir = Path.home() / ".cache" / "smolagents_visual"
    cache_dir.mkdir(parents=True, exist_ok=True)
    source_path = cache_dir / "vision_ocr.swift"
    binary_path = cache_dir / "vision_ocr"
    if not source_path.exists() or source_path.read_text(encoding="utf-8") != APPLE_VISION_OCR_SCRIPT:
        source_path.write_text(APPLE_VISION_OCR_SCRIPT, encoding="utf-8")

    if binary_path.exists() and binary_path.stat().st_mtime >= source_path.stat().st_mtime:
        runtime_cache[cache_key] = str(binary_path)
        return str(binary_path)

    result = subprocess.run(
        ["swiftc", str(source_path), "-O", "-o", str(binary_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None

    runtime_cache[cache_key] = str(binary_path)
    return str(binary_path)


def extract_visible_text_with_apple_vision(image_path: str, image: PIL.Image.Image) -> str:
    binary_path = ensure_apple_vision_ocr_binary()
    if binary_path is None:
        return ""

    temp_paths = []
    seen_lines = set()
    collected_lines = []

    try:
        candidate_paths = [image_path]
        for variant in build_ocr_variants(image)[1:5]:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
                variant.save(handle.name)
                temp_paths.append(handle.name)
                candidate_paths.append(handle.name)

        for candidate_path in candidate_paths:
            result = subprocess.run(
                [binary_path, candidate_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                continue

            lines = [re.sub(r"\s+", " ", line).strip() for line in result.stdout.splitlines()]
            for line in lines:
                if not line:
                    continue
                line_key = line.lower()
                if line_key in seen_lines:
                    continue
                seen_lines.add(line_key)
                collected_lines.append(line)
    finally:
        for temp_path in temp_paths:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    return "\n".join(collected_lines)


def query_is_ocr_relevant(query_lower: str) -> bool:
    return any(keyword in query_lower for keyword in OCR_QUERY_KEYWORDS)


def truncate_query_for_vqa(query: str, max_words: int = 24) -> str:
    words = query.split()
    if len(words) <= max_words:
        return query
    return " ".join(words[:max_words])


def build_ocr_variants(image: PIL.Image.Image) -> list[PIL.Image.Image]:
    width, height = image.size
    variants = [image]

    if width >= 600:
        mid_x = width // 2
        variants.extend(
            [
                image.crop((0, 0, mid_x, height)),
                image.crop((mid_x, 0, width, height)),
            ]
        )
    if height >= 400:
        mid_y = height // 2
        variants.extend(
            [
                image.crop((0, 0, width, mid_y)),
                image.crop((0, mid_y, width, height)),
            ]
        )
    if width >= 900 and height >= 900:
        mid_x = width // 2
        mid_y = height // 2
        variants.extend(
            [
                image.crop((0, 0, mid_x, mid_y)),
                image.crop((mid_x, 0, width, mid_y)),
                image.crop((0, mid_y, mid_x, height)),
                image.crop((mid_x, mid_y, width, height)),
            ]
        )

    prepared_variants = []
    for variant in variants[:6]:
        grayscale = PIL.ImageOps.grayscale(variant)
        contrasted = PIL.ImageOps.autocontrast(grayscale)
        sharpened = PIL.ImageEnhance.Sharpness(contrasted).enhance(1.8)
        if max(sharpened.size) < 1600:
            scale = 1600 / max(sharpened.size)
            sharpened = sharpened.resize(
                (max(1, int(sharpened.width * scale)), max(1, int(sharpened.height * scale))),
                PIL.Image.Resampling.LANCZOS,
            )
        prepared_variants.append(sharpened.convert("RGB"))
    return prepared_variants


def extract_visible_text(image_path: str, image: PIL.Image.Image) -> str:
    import torch

    apple_vision_text = extract_visible_text_with_apple_vision(image_path, image)
    if apple_vision_text.strip():
        return apple_vision_text

    processor, model = get_ocr_components()
    device = get_visual_device()
    extracted_chunks = []
    seen_chunks = set()

    for variant in build_ocr_variants(image):
        pixel_values = processor(images=variant, return_tensors="pt").pixel_values.to(device)
        with torch.inference_mode():
            generated_ids = model.generate(pixel_values, max_new_tokens=128)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        normalized = re.sub(r"\s+", " ", text).strip()
        if normalized and normalized.lower() not in seen_chunks:
            seen_chunks.add(normalized.lower())
            extracted_chunks.append(normalized)

    return "\n".join(extracted_chunks)


def answer_from_ocr_text(query_lower: str, visible_text: str) -> str | None:
    if not visible_text.strip():
        return None

    if any(keyword in query_lower for keyword in OCR_TRANSCRIPTION_KEYWORDS):
        return visible_text

    if any(keyword in query_lower for keyword in OCR_LISTING_KEYWORDS):
        return visible_text

    if any(keyword in query_lower for keyword in OCR_NUMERIC_KEYWORDS):
        numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", visible_text)
        unique_numbers = []
        for number in numbers:
            if number not in unique_numbers:
                unique_numbers.append(number)
        if len(unique_numbers) == 1:
            return unique_numbers[0]

    return None


def process_images_and_text(image_path: str, query: str) -> str:
    import torch

    image_path = materialize_image_path(image_path)
    image = PIL.Image.open(image_path).convert("RGB")
    query_lower = query.lower()
    device = get_visual_device()
    visible_text = ""

    if query_is_ocr_relevant(query_lower):
        try:
            visible_text = extract_visible_text(image_path, image)
        except Exception:
            visible_text = ""
        ocr_answer = answer_from_ocr_text(query_lower, visible_text)
        if ocr_answer:
            return ocr_answer

    if "caption" in query_lower or "describe" in query_lower:
        processor, model = get_caption_components()
        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=80)
        generated = processor.decode(generated_ids[0], skip_special_tokens=True)
        if generated:
            if visible_text:
                return f"{generated}\n\nVisible text:\n{visible_text}"
            return str(generated)

    processor, model = get_vqa_components()
    query_for_vqa = truncate_query_for_vqa(query)
    try:
        encoding = processor(images=image, text=query_for_vqa, return_tensors="pt")
        encoding = {key: value.to(device) for key, value in encoding.items()}
        with torch.inference_mode():
            outputs = model(**encoding)
    except Exception:
        if visible_text:
            return visible_text
        raise
    predicted_idx = outputs.logits.argmax(-1).item()
    answer = model.config.id2label.get(predicted_idx)
    if answer:
        if visible_text:
            return f"{answer}\n\nVisible text:\n{visible_text}"
        return str(answer)

    raise RuntimeError("Visual QA backend returned no usable answer.")


# Function to encode the image
def encode_image(image_path):
    image_path = materialize_image_path(image_path)

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def resize_image(image_path):
    img = PIL.Image.open(image_path)
    width, height = img.size
    img = img.resize((int(width / 2), int(height / 2)))
    new_image_path = f"resized_{image_path}"
    img.save(new_image_path)
    return new_image_path


class VisualQATool(Tool):
    name = "visualizer"
    description = "A tool that can answer questions about attached images."
    inputs = {
        "image_path": {
            "description": "The path to the image on which to answer the question",
            "type": "string",
        },
        "question": {"description": "the question to answer", "type": "string", "nullable": True},
    }
    output_type = "string"

    def forward(self, image_path: str, question: str | None = None) -> str:
        output = ""
        add_note = False
        if not question:
            add_note = True
            question = "Please write a detailed caption for this image."
        try:
            output = process_images_and_text(image_path, question)
        except Exception as e:
            print(e)
            if "Payload Too Large" in str(e):
                new_image_path = resize_image(image_path)
                output = process_images_and_text(new_image_path, question)

        if add_note:
            output = (
                f"You did not provide a particular question, so here is a detailed caption for the image: {output}"
            )

        return output


@tool
def visualizer(image_path: str, question: str | None = None) -> str:
    """A tool that can answer questions about attached images.

    Args:
        image_path: The path to the image on which to answer the question. This should be a local path to downloaded image.
        question: The question to answer.
    """
    add_note = False
    if not question:
        add_note = True
        question = "Please write a detailed caption for this image."
    if not isinstance(image_path, str):
        raise Exception("You should provide at least `image_path` string argument to this tool!")

    try:
        output = process_images_and_text(image_path, question)
    except Exception as e:
        raise Exception(
            "Visual QA request failed "
            f"for caption_model={DEFAULT_VISUAL_CAPTION_MODEL_ID!r} "
            f"vqa_model={DEFAULT_VISUAL_VQA_MODEL_ID!r} "
            f"device={get_visual_device()!r}: {e}"
        ) from e

    if add_note:
        output = f"You did not provide a particular question, so here is a detailed caption for the image: {output}"

    return output
