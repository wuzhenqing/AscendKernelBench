"""Worker-side build, correctness, and timing for one sample.

Runs inside the isolated subprocess started by :func:`eval.eval_sample`.
See docs/guide/evaluation.md for the protocol this module implements.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .compare import (
    compare_candidate_outputs,
    inputs_were_mutated,
    max_abs_diff,
    move_value_to_device,
    resolve_tolerances,
    snapshot_inputs,
    tensor_nbytes,
)
from .eval_result import compiled_result, eval_protocol_metadata, fail_result
from .modes import ACLNN_MODE
from .runtime import (
    npu_device_index,
    npu_runtime_metadata,
    seed_torch,
    torch_dtype_for,
)
from .sol import attach_sol_metadata, task_declared_flops
from .timing import REFRESH_INPUT_BYTES_LIMIT

__all__ = ["eval_sample_on_device", "exec_python_source"]


def exec_python_source(
    source: str,
    filename: str,
    namespace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute ``source`` and return the namespace it populated.

    Args:
        source: Python source text.
        filename: Filename used in compile diagnostics.
        namespace: Optional starting globals. A new dict is used when omitted.

    Returns:
        The executed namespace.

    Raises:
        SyntaxError: If ``source`` does not parse.
        Exception: Any exception raised while executing ``source``.
    """
    populated: dict[str, Any] = {} if namespace is None else namespace
    exec(compile(source, filename, "exec"), populated)
    return populated


def cpu_reference_inputs(raw_inputs: Sequence[Any], torch_mod: Any) -> list[Any]:
    """Cast floating CPU-reference inputs to float32; leave others as-is.

    Args:
        raw_inputs: Values returned by the task ``get_inputs()``.
        torch_mod: The imported ``torch`` module.

    Returns:
        A list suitable for a CPU ``Model`` forward.
    """
    return [
        item.float()
        if isinstance(item, torch_mod.Tensor)
        and torch_mod.is_floating_point(item)
        else item
        for item in raw_inputs
    ]


@dataclass(frozen=True)
class _EvalRequest:
    """Immutable worker inputs for one on-device evaluation."""

    task_py: str
    sample_dir: str
    cmake_arch: str
    hardware_name: str
    device: str
    measure_performance: bool
    seed: int
    num_correct_trials: int
    num_perf_trials: int
    num_warmup: int
    precision: str
    tolerances: dict[str, dict[str, float]]
    excessive_speedup: float
    build_timeout: int
    operator_mode: str
    memory_bandwidth_gbps: float
    peak_tflops: float
    l2_clear_size: int


class _DeviceSession:
    """Stateful evaluation of one compiled sample on the current NPU."""

    def __init__(self, request: _EvalRequest) -> None:
        """Store the request and empty runtime state."""
        self.req = request
        self.sample_path = Path(request.sample_dir)
        self.torch: Any = None
        self.torch_device: Any = None
        self.dtype: Any = None
        self.so_path: Path | None = None
        self.ref_globals: dict[str, object] = {}
        self.Model: Any = None
        self.ModelNew: Any = None
        self.get_init_inputs: Callable[[], Sequence[Any]] | None = None
        self.get_inputs: Callable[[], Sequence[Any]] | None = None
        self.custom_check: Any = None
        self.init_inputs: Sequence[Any] = ()
        self.atol = 1e-4
        self.rtol = 1e-4
        self.new_model: Any = None
        self.ref_model: Any = None
        self.ref_model_cpu: Any = None
        self.ref_mode = "npu"
        self.ref_npu_error: str | None = None
        self.pass_count = 0
        self.max_diff = 0.0
        self.correctness_error = ""
        self.last_new_out: Any = None
        self.runtime: float | None = None
        self.runtime_stats: dict[str, Any] | None = None
        self.ref_runtime: float | None = None
        self.ref_runtime_stats: dict[str, Any] | None = None

    def evaluate(self) -> dict[str, Any]:
        """Run build, correctness, optional timing, and the post-timing check."""
        import torch
        import torch_npu  # noqa: F401

        self.torch = torch
        failure = self._build_and_load()
        if failure is not None:
            return failure
        failure = self._load_python_modules()
        if failure is not None:
            return failure
        self._configure_device()
        failure = self._construct_models()
        if failure is not None:
            return failure
        failure = self._run_correctness_trials()
        if failure is not None:
            return failure
        metadata = self._scored_metadata()
        if self.pass_count != self.req.num_correct_trials:
            metadata["correctness_error"] = self.correctness_error
            return compiled_result(correctness=False, metadata=metadata)
        if self.req.measure_performance:
            self._time_models(metadata)
            failure = self._post_timing_recheck(metadata)
            if failure is not None:
                return failure
        return self._compiled_payload(correctness=True, metadata=metadata)

    def _build_and_load(self) -> dict[str, Any] | None:
        """Build ``libcustom_op.so`` and load it into this process."""
        from .build import BuildError, build_custom_op
        from .loader import LoadError, load_process_local_op

        asc_source = (self.sample_path / "custom_op.asc").read_text(
            encoding="utf-8"
        )
        try:
            self.so_path = build_custom_op(
                asc_source,
                self.sample_path,
                cmake_arch=self.req.cmake_arch,
                timeout_s=self.req.build_timeout,
                operator_mode=self.req.operator_mode,
            )
        except (BuildError, OSError) as exc:
            return fail_result(compilation_error=str(exc))
        try:
            load_process_local_op(self.so_path, asc_source)
        except LoadError as exc:
            return fail_result(
                compiled=True,
                runtime_error=f"shared library load failed: {exc}",
            )
        return None

    def _load_python_modules(self) -> dict[str, Any] | None:
        """Exec the task contract and generated ``ModelNew`` wrapper."""
        model_new_path = self.sample_path / "model_new.py"
        try:
            self.ref_globals = exec_python_source(self.req.task_py, "<task.py>")
            self.Model = self.ref_globals["Model"]
            self.get_init_inputs = self.ref_globals["get_init_inputs"]
            self.get_inputs = self.ref_globals["get_inputs"]
            self.custom_check = self.ref_globals.get("custom_check")
            custom_globals = exec_python_source(
                model_new_path.read_text(encoding="utf-8"),
                str(model_new_path),
                {"__file__": str(model_new_path)},
            )
            self.ModelNew = custom_globals["ModelNew"]
        except Exception as exc:
            return fail_result(
                compiled=True, runtime_error=f"module load failed: {exc!r}"
            )
        task_tolerance = self.ref_globals.get("TOLERANCE")
        self.atol, self.rtol = resolve_tolerances(
            self.req.precision,
            self.req.tolerances,
            task_tolerance if isinstance(task_tolerance, dict) else None,
        )
        return None

    def _configure_device(self) -> None:
        """Select the NPU device and evaluation dtype."""
        self.torch_device = self.torch.device(self.req.device)
        self.torch.npu.set_device(npu_device_index(self.req.device))
        self.dtype = torch_dtype_for(self.req.precision)

    def _require_loaded(self) -> None:
        """Raise if the task contract or ``ModelNew`` is missing after load."""
        missing = [
            name
            for name, value in (
                ("Model", self.Model),
                ("ModelNew", self.ModelNew),
                ("get_init_inputs", self.get_init_inputs),
                ("get_inputs", self.get_inputs),
            )
            if value is None
        ]
        if missing:
            raise RuntimeError(f"eval session missing {missing}")

    def _construct_models(self) -> dict[str, Any] | None:
        """Construct candidate and reference with identical seeded init."""
        self._require_loaded()
        seed_torch(self.req.seed)
        self.init_inputs = self.get_init_inputs()
        try:
            # Identical RNG state for candidate and reference construction:
            # tasks with randomly-initialized parameters are only winnable if
            # ModelNew reproduces the reference weights (KernelBench).
            seed_torch(self.req.seed)
            self.new_model = self.ModelNew(*self.init_inputs).to(
                device=self.torch_device, dtype=self.dtype
            )
            self.new_model.eval()
        except Exception as exc:
            return fail_result(
                compiled=True,
                runtime_error=f"candidate model init failed: {exc!r}",
            )
        # Preferred reference: torch_npu eager on the same device. If that
        # cannot start, fall back to a CPU reference so a correct candidate
        # can surface missing torch_npu coverage.
        try:
            seed_torch(self.req.seed)
            self.ref_model = self.Model(*self.init_inputs).to(
                device=self.torch_device, dtype=self.dtype
            )
            self.ref_model.eval()
        except Exception as exc:
            self.ref_mode = "cpu"
            self.ref_npu_error = repr(exc)
        return None

    def _process_input(self, value: Any) -> Any:
        """Move a tensor to the eval device, casting only floating dtypes."""
        return move_value_to_device(value, self.torch_device, self.dtype)

    def _outputs_ok(self, ref: Any, new: Any) -> bool:
        """Compare outputs with the task override or the default matcher."""
        return compare_candidate_outputs(
            ref,
            new,
            ref_mode=self.ref_mode,
            atol=self.atol,
            rtol=self.rtol,
            custom_check=self.custom_check
            if callable(self.custom_check)
            else None,
        )

    def _run_ref_cpu(self, raw_inputs: Sequence[Any]) -> Any:
        """Run the CPU reference on host float32 copies of ``raw_inputs``."""
        if self.ref_model_cpu is None:
            seed_torch(self.req.seed)
            self.ref_model_cpu = self.Model(*self.init_inputs)
            self.ref_model_cpu.eval()
        return self.ref_model_cpu(
            *cpu_reference_inputs(raw_inputs, self.torch)
        )

    def _run_correctness_trials(self) -> dict[str, Any] | None:
        """Run the seeded correctness trials; return an immediate failure."""
        self._require_loaded()
        seed_torch(self.req.seed)
        trial_seeds = [
            self.torch.randint(0, 2**32 - 1, (1,)).item()
            for _ in range(self.req.num_correct_trials)
        ]
        with self.torch.no_grad():
            for trial, trial_seed in enumerate(trial_seeds):
                failure = self._run_one_correctness_trial(trial, trial_seed)
                if failure is not None:
                    return failure
        return None

    def _run_one_correctness_trial(
        self, trial: int, trial_seed: int
    ) -> dict[str, Any] | None:
        """Execute one correctness trial; return a payload on hard failure."""
        seed_torch(trial_seed)
        raw_inputs = self.get_inputs()
        inputs = [self._process_input(item) for item in raw_inputs]
        seed_torch(trial_seed)
        # torch_npu generators are not guaranteed to be stream-ordered with
        # raw kernel launches; sync so the candidate never reads half-written
        # inputs.
        self.torch.npu.synchronize(device=self.req.device)
        ref_out = self._run_correctness_reference(inputs, raw_inputs, trial)

        # Snapshot after the reference run so the pollution check isolates
        # mutations made by the candidate, not by the reference itself.
        ref_snapshot = snapshot_inputs(inputs)
        try:
            new_out = self.new_model(*inputs)
            self.torch.npu.synchronize(device=self.req.device)
        except Exception as exc:
            return fail_result(
                compiled=True,
                runtime_error=(
                    f"trial {trial}: candidate runtime error: {exc!r}"
                ),
            )
        if inputs_were_mutated(inputs, ref_snapshot):
            return fail_result(
                compiled=True,
                runtime_error=f"trial {trial}: candidate mutated its inputs",
            )
        if self._outputs_ok(ref_out, new_out):
            self.pass_count += 1
        else:
            self.max_diff = max(self.max_diff, max_abs_diff(ref_out, new_out))
            self.correctness_error = (
                f"trial {trial}: output mismatch "
                f"(passed {self.pass_count}/{self.req.num_correct_trials} so far)"
            )
        self.last_new_out = new_out
        return None

    def _run_correctness_reference(
        self, inputs: Sequence[Any], raw_inputs: Sequence[Any], trial: int
    ) -> Any:
        """Run the NPU reference, falling back to CPU for this and later trials.

        Fallback is correctness-only. Timing and the post-timing re-check do
        not switch reference modes (docs/guide/evaluation.md).
        """
        if self.ref_mode != "npu":
            return self._run_ref_cpu(raw_inputs)
        try:
            ref_out = self.ref_model(*inputs)
            self.torch.npu.synchronize(device=self.req.device)
            return ref_out
        except Exception as exc:
            # A trial-order-dependent policy would score the same candidate
            # differently depending on seed order.
            self.ref_mode = "cpu"
            self.ref_npu_error = f"trial {trial}: {exc!r}"
            return self._run_ref_cpu(raw_inputs)

    def _scored_metadata(self) -> dict[str, Any]:
        """Return metadata recorded after the initial correctness trials."""
        metadata: dict[str, Any] = {
            **eval_protocol_metadata(
                hardware_name=self.req.hardware_name,
                precision=self.req.precision,
                seed=self.req.seed,
                num_correct_trials=self.req.num_correct_trials,
                num_warmup=self.req.num_warmup,
                num_perf_trials=self.req.num_perf_trials,
                operator_mode=self.req.operator_mode,
                l2_clear_size=self.req.l2_clear_size,
                atol=self.atol,
                rtol=self.rtol,
            ),
            "reference": self.ref_mode,
            "max_difference": self.max_diff,
            "correctness_passed": self.pass_count,
            "shared_library": str(self.so_path),
            **npu_runtime_metadata(self.req.device),
        }
        if self.ref_npu_error:
            metadata["reference_npu_error"] = self.ref_npu_error
        return metadata

    def _time_models(self, metadata: dict[str, Any]) -> None:
        """Time candidate and NPU reference; attach SOL metadata."""
        from .timing import get_timing_stats, time_execution_with_npu_event

        self._require_loaded()

        def draw_inputs() -> list[Any]:
            """Draw one processed input set for timing."""
            return [self._process_input(item) for item in self.get_inputs()]

        seed_torch(self.req.seed)
        probe_inputs = draw_inputs()
        input_bytes = tensor_nbytes(probe_inputs)
        fresh_per_trial = input_bytes <= REFRESH_INPUT_BYTES_LIMIT
        metadata["timing_fresh_inputs"] = bool(fresh_per_trial)
        perf_box: list[Any] = [probe_inputs]
        prev_box: list[Any] = [None]

        def refresh_inputs() -> None:
            """Replace timing inputs and keep the previous allocation alive."""
            prev_box[0] = perf_box[0]
            perf_box[0] = draw_inputs()

        def timed(fn: Callable[..., Any]) -> list[float]:
            """Time ``fn`` with the evaluation NPU-event protocol."""
            seed_torch(self.req.seed)
            return time_execution_with_npu_event(
                lambda: fn(*perf_box[0]),
                [],
                num_warmup=self.req.num_warmup,
                num_trials=self.req.num_perf_trials,
                device=self.torch_device,
                setup=refresh_inputs if fresh_per_trial else None,
                l2_clear_size=self.req.l2_clear_size,
            )

        try:
            with self.torch.no_grad():
                self.runtime_stats = get_timing_stats(timed(self.new_model))
                self.runtime = self.runtime_stats["mean"]
                if self.ref_mode == "npu":
                    self.ref_runtime_stats = get_timing_stats(
                        timed(self.ref_model)
                    )
                    self.ref_runtime = self.ref_runtime_stats["mean"]
                    self._record_npu_speedup(metadata)
            self._record_sol(metadata, input_bytes)
        except Exception as exc:
            metadata["runtime_error"] = f"timing failed: {exc!r}"

    def _record_npu_speedup(self, metadata: dict[str, Any]) -> None:
        """Record live NPU speedup and the excessive-speedup flag."""
        speedup = self.ref_runtime / self.runtime if self.runtime else 0.0
        metadata["speedup"] = float(f"{speedup:.4g}")
        metadata["excessive_speedup"] = bool(
            speedup > self.req.excessive_speedup
        )

    def _record_sol(self, metadata: dict[str, Any], input_bytes: int) -> None:
        """Attach the roofline bound and SOL score after a timing pass."""
        attach_sol_metadata(
            metadata,
            kernel_ms=self.runtime
            if isinstance(self.runtime, (int, float))
            else None,
            baseline_ms=(
                self.ref_runtime
                if isinstance(self.ref_runtime, (int, float))
                else None
            ),
            bytes_moved=input_bytes + tensor_nbytes(self.last_new_out),
            bandwidth_gbps=float(self.req.memory_bandwidth_gbps),
            flops=task_declared_flops(self.ref_globals),
            peak_tflops=(
                float(self.req.peak_tflops) if self.req.peak_tflops else None
            ),
        )

    def _post_timing_recheck(
        self, metadata: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Fail the sample if a fresh-input re-check mismatches or raises."""
        try:
            seed_torch(self.req.seed + 1)
            recheck_raw = self.get_inputs()
            recheck_inputs = [self._process_input(item) for item in recheck_raw]
            self.torch.npu.synchronize(device=self.req.device)
            with self.torch.no_grad():
                if self.ref_mode == "npu":
                    recheck_ref = self.ref_model(*recheck_inputs)
                    self.torch.npu.synchronize(device=self.req.device)
                else:
                    recheck_ref = self._run_ref_cpu(recheck_raw)
                recheck_new = self.new_model(*recheck_inputs)
                self.torch.npu.synchronize(device=self.req.device)
            if not self._outputs_ok(recheck_ref, recheck_new):
                metadata["correctness_error"] = (
                    "post-timing fresh-input re-check failed: outputs are not "
                    "a pure function of current inputs (caching or state drift)"
                )
                return self._compiled_payload(
                    correctness=False, metadata=metadata
                )
        except Exception as exc:
            metadata["runtime_error"] = f"post-timing re-check failed: {exc!r}"
            return self._compiled_payload(correctness=False, metadata=metadata)
        return None

    def _compiled_payload(
        self, *, correctness: bool, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Return a post-build payload using the session timing fields."""
        return compiled_result(
            correctness=correctness,
            metadata=metadata,
            runtime=self.runtime,
            runtime_stats=self.runtime_stats,
            ref_runtime=self.ref_runtime,
            ref_runtime_stats=self.ref_runtime_stats,
        )


def eval_sample_on_device(
    *,
    task_py: str,
    sample_dir: str,
    cmake_arch: str,
    hardware_name: str,
    device: str,
    measure_performance: bool,
    seed: int,
    num_correct_trials: int,
    num_perf_trials: int,
    num_warmup: int,
    precision: str,
    tolerances: dict[str, dict[str, float]],
    excessive_speedup: float,
    build_timeout: int,
    operator_mode: str = ACLNN_MODE,
    memory_bandwidth_gbps: float = 0.0,
    peak_tflops: float = 0.0,
    l2_clear_size: int = 256 * 1024 * 1024,
) -> dict[str, Any]:
    """Build, check correctness, and time a sample in the current process.

    Args:
        task_py: Reference task source (``Model``, ``get_inputs``, ...).
        sample_dir: Directory with ``custom_op.asc`` and ``model_new.py``.
        cmake_arch: Value passed to ``CMAKE_ASC_ARCHITECTURES``.
        hardware_name: Profile name recorded in result metadata.
        device: Runtime device string, for example ``npu:0``.
        measure_performance: When False, skip timing and post-timing check.
        seed: Base RNG seed for init, correctness, and timing.
        num_correct_trials: Number of seeded correctness trials.
        num_perf_trials: Retained NPU-event measurements per model.
        num_warmup: Warmup calls before measured trials.
        precision: Floating dtype key (``fp32``, ``fp16``, or ``bf16``).
        tolerances: Per-precision ``atol`` / ``rtol`` mapping.
        excessive_speedup: Flag speedups strictly above this ratio.
        build_timeout: Seconds allowed for CMake configure and for build.
        operator_mode: Compilation mode; only ``aclnn`` is implemented.
        memory_bandwidth_gbps: Profile bandwidth for the roofline SOL bound.
        peak_tflops: Optional peak TFLOPS (unused without a FLOP count).
        l2_clear_size: Bytes allocated to flush L2 before each timed call.

    Returns:
        KernelBench-compatible per-sample result dict.
    """
    return _DeviceSession(
        _EvalRequest(
            task_py=task_py,
            sample_dir=sample_dir,
            cmake_arch=cmake_arch,
            hardware_name=hardware_name,
            device=device,
            measure_performance=measure_performance,
            seed=seed,
            num_correct_trials=num_correct_trials,
            num_perf_trials=num_perf_trials,
            num_warmup=num_warmup,
            precision=precision,
            tolerances=tolerances,
            excessive_speedup=excessive_speedup,
            build_timeout=build_timeout,
            operator_mode=operator_mode,
            memory_bandwidth_gbps=memory_bandwidth_gbps,
            peak_tflops=peak_tflops,
            l2_clear_size=l2_clear_size,
        )
    ).evaluate()
