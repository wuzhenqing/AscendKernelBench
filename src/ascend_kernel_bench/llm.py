"""OpenAI-compatible LLM client with pydantic-structured output.

See docs/deploy_llm_service.md for the endpoint and response contracts.

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
    """Remove one outer fenced-code-block wrapper if the model added it."""
    match = _FENCE_EDGE_RE.match(value)
    body = match.group("body") if match else value
    # Some models emit a bare filename line ("custom_op.asc") as the first line.
    lines = body.split("\n")
    if lines and re.fullmatch(
        r"\s*(custom_op\.asc|model_new\.py)\s*", lines[0]
    ):
        body = "\n".join(lines[1:])
    return body


class AscendCGeneration(BaseModel):
    """The two code deliverables of one generation (docs/task_authoring.md)."""

    custom_op_asc: str = Field(
        description=(
            "Complete self-contained Ascend C source file custom_op.asc: "
            "kernel class, __global__ __vector__ kernel, host launch wrapper "
            "taking at::Tensor, and a process-local torch.library binding "
            "(TORCH_LIBRARY(custom_op, ...) and "
            "TORCH_LIBRARY_IMPL(custom_op, PrivateUse1, ...)). "
            "Raw file content only, no markdown fences. Do not use pybind11."
        )
    )
    model_new_py: str = Field(
        description=(
            "Python source of model_new.py defining class ModelNew with the "
            "same __init__ and forward signatures as the reference Model, "
            "calling the evaluator-loaded operator via torch.ops.custom_op. "
            "Raw file content only, no markdown fences."
        )
    )

    @field_validator("custom_op_asc", "model_new_py", mode="before")
    @classmethod
    def _strip_markdown_fence(cls, value: str) -> str:
        """Drop an outer markdown fence or leading filename line."""
        return _strip_fence(value) if isinstance(value, str) else value


@dataclass(frozen=True)
class GenerationResult:
    """One LLM response after structured or fenced-block extraction."""

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


def _usage_dict(response: object) -> dict:
    """Return ``response.usage`` as a dict, or ``{}`` when absent."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    dump = getattr(usage, "model_dump", None)
    return dump() if callable(dump) else {}


_MIN_ASC_KERNEL_MARKERS = ("__global__", "__vector__")
_MIN_ASC_BINDING_MARKERS = ("TORCH_LIBRARY", "TORCH_LIBRARY_IMPL")
_MIN_PY_MARKERS = ("class ModelNew", "torch.ops.custom_op")


def validate_generation(gen: AscendCGeneration) -> list[str]:
    """Sanity-check that the fields carry real file content.

    Args:
        gen: Parsed deliverables.

    Returns:
        Human-readable problems; empty means the fields look complete.
    """
    problems = []
    for marker in _MIN_ASC_KERNEL_MARKERS:
        if marker not in gen.custom_op_asc:
            problems.append(f"custom_op_asc missing {marker!r}")
    for marker in _MIN_ASC_BINDING_MARKERS:
        if marker not in gen.custom_op_asc:
            problems.append(f"custom_op_asc missing {marker!r}")
    if "PYBIND11_MODULE" in gen.custom_op_asc:
        problems.append("custom_op_asc must not use PYBIND11_MODULE")
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
        """Create a client for one generation model.

        Args:
            model: Served model name passed to the OpenAI-compatible API.
            base_url: Endpoint URL; defaults to ``OPENAI_BASE_URL``.
            api_key: Credential; defaults to ``OPENAI_API_KEY``.
            temperature: Sampling temperature.
            max_tokens: Completion token budget.
            timeout: Request timeout in seconds.
        """
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
        """Generate one sample via structured parse, else fenced blocks.

        The response is validated for real file content; an invalid answer is
        retried once with an explicit correction reminder.

        Args:
            prompt: User prompt (the structured-output note is appended).
            system: Optional system message.
            max_retries: Extra attempts after the first failed parse or
                validation. Default is one retry.

        Returns:
            Parsed deliverables plus the raw response text.

        Raises:
            ValueError: If every attempt fails validation or the endpoint
                returns no usable fenced blocks.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append(
            {"role": "user", "content": prompt + STRUCTURED_OUTPUT_NOTE}
        )

        last_error: Exception | None = None
        last_raw = ""
        for attempt in range(max_retries + 1):
            if attempt > 0:
                _append_retry_turn(messages, last_raw)
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
        """Request one completion; fall back to fenced-block extraction."""
        try:
            return self._parse_structured(messages)
        except Exception:
            # Endpoint may not support structured outputs: plain completion +
            # fenced-block extraction, validated by the same pydantic model.
            return self._parse_fenced(messages)

    def _parse_structured(self, messages: list[dict]) -> GenerationResult:
        """Parse a structured-output response into the two deliverables."""
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
        return GenerationResult(
            parsed,
            response.choices[0].message.content or "",
            self.model,
            _usage_dict(response),
        )

    def _parse_fenced(self, messages: list[dict]) -> GenerationResult:
        """Complete without a schema and extract fenced code blocks."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        raw = response.choices[0].message.content or ""
        return GenerationResult(
            extract_generation(raw), raw, self.model, _usage_dict(response)
        )


_RETRY_REMINDER = (
    "Your previous answer did not contain the "
    "required file contents. Return the COMPLETE "
    "custom_op.asc source in `custom_op_asc` and "
    "the COMPLETE model_new.py source in "
    "`model_new_py` — full code, no placeholders."
)


def _append_retry_turn(messages: list[dict], last_raw: str) -> None:
    """Append the previous answer and a correction reminder."""
    if last_raw:
        messages.append({"role": "assistant", "content": last_raw})
    messages.append({"role": "user", "content": _RETRY_REMINDER})


_FENCE_RE = re.compile(
    r"```(?P<tag>[A-Za-z0-9_+.-]*)\s*\n(?P<body>.*?)```", re.DOTALL
)


def extract_generation(text: str) -> AscendCGeneration:
    """Extract the two deliverables from fenced code blocks.

    Recognised layouts (in priority order):
    1. Blocks tagged with filenames (```custom_op.asc / ```model_new.py).
    2. A C++-tagged block (cpp/c++/asc) then a python-tagged block.
    3. The first two fenced blocks, Ascend C first.

    Args:
        text: Raw model response.

    Returns:
        Parsed ``custom_op_asc`` and ``model_new_py`` fields.

    Raises:
        ValueError: If no usable pair of fenced blocks is found.
    """
    blocks = list(_FENCE_RE.finditer(text))
    if not blocks:
        raise ValueError("no fenced code blocks found in model response")
    asc_src, py_src = _pair_fenced_blocks(blocks)
    if asc_src is None or py_src is None:
        raise ValueError(
            "could not identify custom_op.asc and model_new.py blocks"
        )
    return AscendCGeneration(custom_op_asc=asc_src, model_new_py=py_src)


def _pair_fenced_blocks(
    blocks: list[re.Match[str]],
) -> tuple[str | None, str | None]:
    """Pick Ascend C and Python bodies from fenced blocks.

    Filename tags win, then language tags, then first-two-block order.
    """
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
    if asc_src is None and blocks:
        asc_src = blocks[0].group("body")
    if py_src is None and len(blocks) >= 2:
        py_src = blocks[1].group("body")
    return asc_src, py_src
