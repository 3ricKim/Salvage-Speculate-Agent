import os
from pathlib import Path

from smolagents import Tool
from smolagents.models import Model


class TextInspectorTool(Tool):
    name = "inspect_file_as_text"
    description = """
You cannot load files yourself: instead call this tool to read a file as markdown text and ask questions about it.
This tool handles the following file extensions: [".html", ".htm", ".xlsx", ".pptx", ".wav", ".mp3", ".m4a", ".flac", ".pdf", ".docx", ".json", ".jsonl", ".jsonld"], and all other types of text files. IT DOES NOT HANDLE IMAGES."""

    inputs = {
        "file_path": {
            "description": "The path to the file you want to read as text. Must be a '.something' file, like '.pdf'. If it is an image, use the visualizer tool instead! DO NOT use this tool for an HTML webpage: use the web_search tool instead!",
            "type": "string",
        },
        "question": {
            "description": "[Optional]: Your question, as a natural language sentence. Provide as much context as possible. Do not pass this parameter if you just want to directly return the content of the file.",
            "type": "string",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, model: Model = None, text_limit: int = 100000):
        super().__init__()
        self.model = model
        self.text_limit = text_limit
        from .mdconvert import MarkdownConverter

        self.md_converter = MarkdownConverter()

    def _resolve_file_path(self, file_path: str) -> str:
        if file_path.startswith(("http://", "https://", "file://")):
            return file_path

        candidate = Path(file_path).expanduser()
        if candidate.is_file():
            return str(candidate.resolve())

        script_root = Path(__file__).resolve().parent.parent
        search_roots = [
            Path.cwd(),
            script_root,
            Path.cwd() / "downloads_folder",
            Path.cwd() / "downloads",
            script_root / "downloads_folder",
            script_root / "downloads",
        ]
        basename = candidate.name
        matches: list[Path] = []
        for root in search_roots:
            if not root.exists():
                continue
            direct_match = root / basename
            if direct_match.is_file():
                matches.append(direct_match)
                continue
            matches.extend(path for path in root.rglob(basename) if path.is_file())

        # Prefer the newest downloaded file if there are multiple matches.
        if matches:
            best_match = max(matches, key=lambda path: path.stat().st_mtime)
            return str(best_match.resolve())

        return file_path

    def forward_initial_exam_mode(self, file_path, question):
        from smolagents.models import MessageRole

        file_path = self._resolve_file_path(file_path)
        result = self.md_converter.convert(file_path)

        if file_path[-4:] in [".png", ".jpg"]:
            raise Exception("Cannot use inspect_file_as_text tool with images: use visualizer instead!")

        if ".zip" in file_path:
            return result.text_content

        if not question:
            return result.text_content

        if len(result.text_content) < 4000:
            return "Document content: " + result.text_content

        messages = [
            {
                "role": MessageRole.SYSTEM,
                "content": [
                    {
                        "type": "text",
                        "text": "Here is a file:\n### "
                        + str(result.title)
                        + "\n\n"
                        + result.text_content[: self.text_limit],
                    }
                ],
            },
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": "Now please write a short, 5 sentence caption for this document, that could help someone asking this question: "
                        + question
                        + "\n\nDon't answer the question yourself! Just provide useful notes on the document",
                    }
                ],
            },
        ]
        return self.model(messages).content

    def forward(self, file_path, question: str | None = None) -> str:
        from smolagents.models import MessageRole

        file_path = self._resolve_file_path(file_path)
        result = self.md_converter.convert(file_path)

        if file_path[-4:] in [".png", ".jpg"]:
            raise Exception("Cannot use inspect_file_as_text tool with images: use visualizer instead!")

        if ".zip" in file_path:
            return result.text_content

        if not question:
            return result.text_content

        messages = [
            {
                "role": MessageRole.SYSTEM,
                "content": [
                    {
                        "type": "text",
                        "text": "You will have to write a short caption for this file, then answer this question:"
                        + question,
                    }
                ],
            },
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": "Here is the complete file:\n### "
                        + str(result.title)
                        + "\n\n"
                        + result.text_content[: self.text_limit],
                    }
                ],
            },
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": "Now answer the question below. Use these three headings: '1. Short answer', '2. Extremely detailed answer', '3. Additional Context on the document and question asked'."
                        + question,
                    }
                ],
            },
        ]
        return self.model(messages).content
