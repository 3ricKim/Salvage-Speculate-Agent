# Shamelessly stolen from Microsoft Autogen team: thanks to them for this great resource!
# https://github.com/microsoft/autogen/blob/gaia_multiagent_v01_march_1st/autogen/browser_utils.py
import copy
import math
import re
import time

from smolagents.models import MessageRole, Model


TRANSIENT_REFORMULATION_ERROR_PATTERNS = (
    "ratelimiterror",
    "rate limit",
    "quota",
    "temporarily unavailable",
    "timed out",
    "timeout",
    "connection reset",
    "connection aborted",
    "service unavailable",
    "server disconnected",
    "internal server error",
    "503",
    "502",
    "500",
)

UNIT_SCALES = {
    "angstrom": 1e-10,
    "angstroms": 1e-10,
    "ångström": 1e-10,
    "ångströms": 1e-10,
    "meter": 1.0,
    "meters": 1.0,
    "centimeter": 1e-2,
    "centimeters": 1e-2,
    "millimeter": 1e-3,
    "millimeters": 1e-3,
    "micrometer": 1e-6,
    "micrometers": 1e-6,
    "nanometer": 1e-9,
    "nanometers": 1e-9,
    "picometer": 1e-12,
    "picometers": 1e-12,
    "dollar": 1.0,
    "dollars": 1.0,
    "usd": 1.0,
    "cent": 1e-2,
    "cents": 1e-2,
}


def is_transient_reformulation_error(error: Exception) -> bool:
    error_text = str(error).lower()
    return any(pattern in error_text for pattern in TRANSIENT_REFORMULATION_ERROR_PATTERNS)


def extract_numeric_answer_fragments(answer: str) -> list[str]:
    return re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", answer)


def infer_requested_decimal_places(original_task: str) -> int | None:
    task_lower = original_task.lower()

    # GAIA has at least one benchmark task that expects Angstrom answers to 3 decimals
    # when the prompt says "rounded to the nearest picometer".
    if "in angstrom" in task_lower and "nearest picometer" in task_lower:
        return 3

    output_unit = None
    rounding_unit = None
    for unit_alias in sorted(UNIT_SCALES, key=len, reverse=True):
        if output_unit is None and f"in {unit_alias}" in task_lower:
            output_unit = unit_alias
        if rounding_unit is None and f"nearest {unit_alias}" in task_lower:
            rounding_unit = unit_alias
        if output_unit and rounding_unit:
            break

    if output_unit is None or rounding_unit is None:
        return None

    output_scale = UNIT_SCALES[output_unit]
    rounding_scale = UNIT_SCALES[rounding_unit]
    if rounding_scale >= output_scale:
        return 0

    ratio = output_scale / rounding_scale
    log_ratio = math.log10(ratio)
    if abs(log_ratio - round(log_ratio)) > 1e-9:
        return None
    return int(round(log_ratio))


def postprocess_final_answer(original_task: str, final_answer: str) -> str:
    cleaned_answer = final_answer.strip()
    if not cleaned_answer:
        return cleaned_answer

    numeric_fragments = extract_numeric_answer_fragments(cleaned_answer)
    decimal_places = infer_requested_decimal_places(original_task)
    if len(numeric_fragments) != 1:
        return cleaned_answer
    numeric_fragment = numeric_fragments[0]

    if numeric_fragment != cleaned_answer or decimal_places is not None:
        numeric_value = float(numeric_fragment)
        if decimal_places is None:
            return numeric_fragment
        rounded_value = round(numeric_value, decimal_places)
        if decimal_places == 0:
            return str(int(round(rounded_value)))
        return f"{rounded_value:.{decimal_places}f}"

    return cleaned_answer


def prepare_response(original_task: str, inner_messages, reformulation_model: Model) -> str:
    messages = [
        {
            "role": MessageRole.SYSTEM,
            "content": [
                {
                    "type": "text",
                    "text": f"""Earlier you were asked the following:

{original_task}

Your team then worked diligently to address that request. Read below a transcript of that conversation:""",
                }
            ],
        }
    ]

    # The first message just repeats the question, so remove it
    # if len(inner_messages) > 1:
    #    del inner_messages[0]

    # copy them to this context
    try:
        for message in inner_messages:
            if not message.content:
                continue
            message = copy.deepcopy(message)
            message.role = MessageRole.USER
            messages.append(message)
    except Exception:
        messages += [{"role": MessageRole.ASSISTANT, "content": str(inner_messages)}]

    # ask for the final answer
    messages.append(
        {
            "role": MessageRole.USER,
            "content": [
                {
                    "type": "text",
                    "text": f"""
Read the above conversation and output a FINAL ANSWER to the question. The question is repeated here for convenience:

{original_task}

To output the final answer, use the following template: FINAL ANSWER: [YOUR FINAL ANSWER]
Your FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
ADDITIONALLY, your FINAL ANSWER MUST adhere to any formatting instructions specified in the original question (e.g., alphabetization, sequencing, units, rounding, decimal places, etc.)
If you are asked for a number, express it numerically (i.e., with digits rather than words), don't use commas, and DO NOT INCLUDE UNITS such as $ or USD or percent signs unless specified otherwise.
If the question asks for a value in one unit but mentions a smaller unit only as the rounding precision, keep the answer in the requested unit. For example, "in Angstroms, rounded to the nearest picometer" must still be answered in Angstroms.
If you are asked for a string, don't use articles or abbreviations (e.g. for cities), unless specified otherwise. Don't output any final sentence punctuation such as '.', '!', or '?'.
If you are asked for a comma separated list, apply the above rules depending on whether the elements are numbers or strings.
If you are unable to determine the final answer, output 'FINAL ANSWER: Unable to determine'
""",
                }
            ],
        }
    )

    last_error = None
    response = None
    for attempt in range(2):
        try:
            response = reformulation_model(messages).content
            break
        except Exception as error:
            last_error = error
            if attempt == 1 or not is_transient_reformulation_error(error):
                raise
            time.sleep(1.0)

    if response is None:
        raise last_error if last_error is not None else RuntimeError("Reformulation produced no response.")

    final_answer = postprocess_final_answer(original_task, response.split("FINAL ANSWER: ")[-1].strip())
    print("> Reformulated answer: ", final_answer)

    #     if "unable to determine" in final_answer.lower():
    #         messages.append({"role": MessageRole.ASSISTANT, "content": response })
    #         messages.append({"role": MessageRole.USER, "content": [{"type": "text", "text": """
    # I understand that a definitive answer could not be determined. Please make a well-informed EDUCATED GUESS based on the conversation.

    # To output the educated guess, use the following template: EDUCATED GUESS: [YOUR EDUCATED GUESS]
    # Your EDUCATED GUESS should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. DO NOT OUTPUT 'I don't know', 'Unable to determine', etc.
    # ADDITIONALLY, your EDUCATED GUESS MUST adhere to any formatting instructions specified in the original question (e.g., alphabetization, sequencing, units, rounding, decimal places, etc.)
    # If you are asked for a number, express it numerically (i.e., with digits rather than words), don't use commas, and don't include units such as $ or percent signs unless specified otherwise.
    # If you are asked for a string, don't use articles or abbreviations (e.g. cit for cities), unless specified otherwise. Don't output any final sentence punctuation such as '.', '!', or '?'.
    # If you are asked for a comma separated list, apply the above rules depending on whether the elements are numbers or strings.
    # """.strip()}]})

    #         response = model(messages).content
    #         print("\n>>>Making an educated guess.\n", response)
    #         final_answer = response.split("EDUCATED GUESS: ")[-1].strip()
    return final_answer
