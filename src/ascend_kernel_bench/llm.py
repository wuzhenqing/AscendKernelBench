"""OpenAI-compatible LLM client with pydantic-structured output (README 3.9).

The endpoint is whatever ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` point to.
Structured generation uses the OpenAI ``parse`` API with a pydantic model so
the two deliverables (``custom_op_asc``, ``model_new_py``) arrive as typed
fields — no hand-rolled field parsing. Endpoints without structured-output
support fall back to fenced-code-block extraction, still validated by the
same pydantic model.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

_FENCE_EDGE_RE = re.compile(
    r"^\s*```[A-Za-z0-9_+.-]*\s*\n(?P<body>.*?)\n?\s*```\s*$", re.DOTALL
)


def _strip_fence(value: str) -> str:
    """Remove a single outer fenced-code-block wrapper if the model added one."""
    match = _FENCE_EDGE_RE.match(value)
    body = match.group("body") if match else value
    # Some models emit a bare filename line ("custom_op.asc") as the first line.
    lines = body.split("\n")
    if lines and re.fullmatch(r"\s*(custom_op\.asc|model_new\.py)\s*", lines[0]):
        body = "\n".join(lines[1:])
    return body


class AscendCGeneration(BaseModel):
    """The two code deliverables of one generation (README 4.3)."""

    custom_op_asc: str = Field(
        description=(
            "Complete self-contained Ascend C source file custom_op.asc: "
            "kernel class, __global__ __vector__ kernel, host launch wrapper "
            "taking at::Tensor, and PYBIND11_MODULE(custom_op, m) binding. "
            "Raw file content only, no markdown fences."
        )
    )
    model_new_py: str = Field(
        description=(
            "Python source of model_new.py defining class ModelNew with the "
            "same __init__ and forward signatures as the reference Model, "
            "internally importing custom_op and calling the compiled operator. "
            "Raw file content only, no markdown fences."
        )
    )

    @field_validator("custom_op_asc", "model_new_py", mode="before")
    @classmethod
    def _strip_markdown_fence(cls, value: str) -> str:
        return _strip_fence(value) if isinstance(value, str) else value


@dataclass(frozen=True)
class GenerationResult:
    generation: AscendCGeneration
    raw_text: str
    model: str
    usage: dict


STRUCTURED_OUTPUT_NOTE = (
    "\n\n## Response Format\n\n"
    "Respond through the structured JSON schema: put the FULL raw content of "
    "the Ascend C file into the `custom_op_asc` field and the FULL raw content "
    "of the Python file into the `model_new_py` field. The field values are "
    "the files themselves (every line of code), NOT filenames, NOT summaries, "
    "and without markdown fences."
)

_MIN_ASC_MARKERS = ("PYBIND11_MODULE", "__global__", "__vector__")
_MIN_PY_MARKERS = ("class ModelNew",)


def validate_generation(gen: AscendCGeneration) -> list[str]:
    """Sanity-check that the fields carry real file content."""
    problems = []
    for marker in _MIN_ASC_MARKERS:
        if marker not in gen.custom_op_asc:
            problems.append(f"custom_op_asc missing {marker!r}")
    for marker in _MIN_PY_MARKERS:
        if marker not in gen.model_new_py:
            problems.append(f"model_new_py missing {marker!r}")
    return problems


class LLMClient:
    """Thin OpenAI-compatible client for Ascend C operator generation."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 16384,
        timeout: float = 600.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            timeout=timeout,
        )

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_retries: int = 1,
    ) -> GenerationResult:
        """Generate one structured sample; parse with pydantic, fall back to fences.

        The response is validated for real file content; an invalid answer is
        retried once with an explicit correction reminder.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt + STRUCTURED_OUTPUT_NOTE})

        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            if attempt > 0:
                if last_raw:
                    messages.append({"role": "assistant", "content": last_raw})
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous answer did not contain the required file "
                        "contents. Return the COMPLETE custom_op.asc source in "
                        "`custom_op_asc` and the COMPLETE model_new.py source in "
                        "`model_new_py` — full code, no placeholders."
                    ),
                })
            try:
                result = self._generate_once(messages)
                problems = validate_generation(result.generation)
                if not problems:
                    return result
                last_error = ValueError("; ".join(problems))
                last_raw = result.raw_text
            except Exception as exc:
                last_error = exc
                last_raw = ""
        raise ValueError(f"generation failed validation: {last_error}")

    def _generate_once(self, messages: list[dict]) -> GenerationResult:
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=AscendCGeneration,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            parsed = response.choices[0].message.parsed
            if parsed is None:
                raise ValueError("structured parse returned None")
            raw = response.choices[0].message.content or ""
            usage = (
                response.usage.model_dump() if response.usage else {}
            )
            return GenerationResult(parsed, raw, self.model, usage)
        except Exception:
            # Endpoint may not support structured outputs: plain completion +
            # fenced-block extraction, validated by the same pydantic model.
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            raw = response.choices[0].message.content or ""
            usage = (
                response.usage.model_dump() if response.usage else {}
            )
            return GenerationResult(
                extract_generation(raw), raw, self.model, usage
            )


_FENCE_RE = re.compile(
    r"```(?P<tag>[A-Za-z0-9_+.-]*)\s*\n(?P<body>.*?)```", re.DOTALL
)


def extract_generation(text: str) -> AscendCGeneration:
    """Extract the two deliverables from fenced code blocks.

    Recognised layouts (in priority order):
    1. Blocks tagged with filenames (```custom_op.asc / ```model_new.py).
    2. A C++-tagged block (cpp/c++/asc) then a python-tagged block.
    3. The first two fenced blocks, Ascend C first.
    """
    blocks = list(_FENCE_RE.finditer(text))
    if not blocks:
        raise ValueError("no fenced code blocks found in model response")

    asc_src: str | None = None
    py_src: str | None = None
    for block in blocks:
        tag = block.group("tag").lower()
        body = block.group("body")
        if "custom_op.asc" in tag or "custom_op_asc" in tag:
            asc_src = body
        elif "model_new.py" in tag or "model_new_py" in tag:
            py_src = body
    if asc_src is None or py_src is None:
        for block in blocks:
            tag = block.group("tag").lower()
            body = block.group("body")
            if asc_src is None and tag in {"cpp", "c++", "asc", "c"}:
                asc_src = body
            elif py_src is None and tag in {"python", "py"}:
                py_src = body
    if asc_src is None and len(blocks) >= 1:
        asc_src = blocks[0].group("body")
    if py_src is None and len(blocks) >= 2:
        py_src = blocks[1].group("body")
    if asc_src is None or py_src is None:
        raise ValueError("could not identify custom_op.asc and model_new.py blocks")
    return AscendCGeneration(custom_op_asc=asc_src, model_new_py=py_src)
