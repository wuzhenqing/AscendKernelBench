"""Worker-side build, correctness, and timing for one sample.

Runs inside the isolated subprocess started by :func:`eval.eval_sample`.
See docs/guide/evaluation.md for the protocol this module implements.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .build import BuildError, LoadError, build_custom_op, load_custom_op
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
from .runtime import (
    npu_device_index,
    npu_runtime_metadata,
    seed_torch,
    torch_dtype_for,
)
from .sol import attach_sol_metadata, task_declared_flops
from .timing import (
    REFRESH_INPUT_BYTES_LIMIT,
    get_timing_stats,
    time_execution_with_npu_event,
)

__all__ = ["eval_sample_on_device", "exec_python_source"]


def exec_python_source(
    source: str,
    filename: str,
    namespace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute ``source`` and return the namespace it populated."""
    populated: dict[str, Any] = {} if namespace is None else namespace
    exec(compile(source, filename, "exec"), populated)
    return populated


def cpu_reference_inputs(
    raw_inputs: Sequence[Any], torch_mod: Any
) -> list[Any]:
    """Cast floating CPU-reference inputs to float32; leave others as-is."""
    return [
        item.float()
        if isinstance(item, torch_mod.Tensor)
        and torch_mod.is_floating_point(item)
        else item
        for item in raw_inputs
    ]


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
    memory_bandwidth_gbps: float = 0.0,
    peak_tflops: float = 0.0,
    l2_clear_size: int = 256 * 1024 * 1024,
) -> dict[str, Any]:
    """Build, check correctness, and time a sample in the current process."""
    import torch
    import torch_npu  # noqa: F401

    sample_path = Path(sample_dir)
    asc_source = (sample_path / "custom_op.asc").read_text(encoding="utf-8")
    try:
        so_path = build_custom_op(
            asc_source,
            sample_path,
            cmake_arch=cmake_arch,
            timeout_s=build_timeout,
        )
        load_custom_op(so_path, asc_source)
    except (BuildError, OSError) as exc:
        return fail_result(compilation_error=str(exc))
    except LoadError as exc:
        return fail_result(
            compiled=True,
            runtime_error=f"shared library load failed: {exc}",
        )

    try:
        ref_globals = exec_python_source(task_py, "<task.py>")
        model_cls = ref_globals["Model"]
        get_init_inputs = ref_globals["get_init_inputs"]
        get_inputs = ref_globals["get_inputs"]
        custom_check = ref_globals.get("custom_check")
        model_new_path = sample_path / "model_new.py"
        custom_globals = exec_python_source(
            model_new_path.read_text(encoding="utf-8"),
            str(model_new_path),
            {"__file__": str(model_new_path)},
        )
        model_new_cls = custom_globals["ModelNew"]
    except Exception as exc:
        return fail_result(
            compiled=True, runtime_error=f"module load failed: {exc!r}"
        )

    task_tolerance = ref_globals.get("TOLERANCE")
    atol, rtol = resolve_tolerances(
        precision,
        tolerances,
        task_tolerance if isinstance(task_tolerance, dict) else None,
    )

    torch_device = torch.device(device)
    torch.npu.set_device(npu_device_index(device))
    dtype = torch_dtype_for(precision)

    seed_torch(seed)
    init_inputs = get_init_inputs()
    try:
        seed_torch(seed)
        new_model = model_new_cls(*init_inputs).to(
            device=torch_device, dtype=dtype
        )
        new_model.eval()
    except Exception as exc:
        return fail_result(
            compiled=True,
            runtime_error=f"candidate model init failed: {exc!r}",
        )

    ref_mode = "npu"
    ref_npu_error: str | None = None
    ref_model = None
    ref_model_cpu = None
    try:
        seed_torch(seed)
        ref_model = model_cls(*init_inputs).to(device=torch_device, dtype=dtype)
        ref_model.eval()
    except Exception as exc:
        ref_mode = "cpu"
        ref_npu_error = repr(exc)

    def process_input(value: Any) -> Any:
        return move_value_to_device(value, torch_device, dtype)

    def outputs_ok(ref: Any, new: Any) -> bool:
        return compare_candidate_outputs(
            ref,
            new,
            ref_mode=ref_mode,
            atol=atol,
            rtol=rtol,
            custom_check=custom_check if callable(custom_check) else None,
        )

    def run_ref_cpu(raw_inputs: Sequence[Any]) -> Any:
        nonlocal ref_model_cpu
        if ref_model_cpu is None:
            seed_torch(seed)
            ref_model_cpu = model_cls(*init_inputs)
            ref_model_cpu.eval()
        return ref_model_cpu(*cpu_reference_inputs(raw_inputs, torch))

    def run_reference(
        inputs: Sequence[Any], raw_inputs: Sequence[Any], trial: int
    ) -> Any:
        nonlocal ref_mode, ref_npu_error
        if ref_mode != "npu":
            return run_ref_cpu(raw_inputs)
        try:
            ref_out = ref_model(*inputs)
            torch.npu.synchronize(device=device)
            return ref_out
        except Exception as exc:
            ref_mode = "cpu"
            ref_npu_error = f"trial {trial}: {exc!r}"
            return run_ref_cpu(raw_inputs)

    pass_count = 0
    max_diff = 0.0
    correctness_error = ""
    last_new_out: Any = None
    seed_torch(seed)
    trial_seeds = [
        torch.randint(0, 2**32 - 1, (1,)).item()
        for _ in range(num_correct_trials)
    ]
    with torch.no_grad():
        for trial, trial_seed in enumerate(trial_seeds):
            seed_torch(trial_seed)
            raw_inputs = get_inputs()
            inputs = [process_input(item) for item in raw_inputs]
            seed_torch(trial_seed)
            torch.npu.synchronize(device=device)
            ref_out = run_reference(inputs, raw_inputs, trial)
            ref_snapshot = snapshot_inputs(inputs)
            try:
                new_out = new_model(*inputs)
                torch.npu.synchronize(device=device)
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
                    runtime_error=(
                        f"trial {trial}: candidate mutated its inputs"
                    ),
                )
            if outputs_ok(ref_out, new_out):
                pass_count += 1
            else:
                max_diff = max(max_diff, max_abs_diff(ref_out, new_out))
                correctness_error = (
                    f"trial {trial}: output mismatch "
                    f"(passed {pass_count}/{num_correct_trials} so far)"
                )
            last_new_out = new_out

    metadata: dict[str, Any] = {
        **eval_protocol_metadata(
            hardware_name=hardware_name,
            precision=precision,
            seed=seed,
            num_correct_trials=num_correct_trials,
            num_warmup=num_warmup,
            num_perf_trials=num_perf_trials,
            l2_clear_size=l2_clear_size,
            atol=atol,
            rtol=rtol,
        ),
        "reference": ref_mode,
        "max_difference": max_diff,
        "correctness_passed": pass_count,
        "shared_library": str(so_path),
        **npu_runtime_metadata(device),
    }
    if ref_npu_error:
        metadata["reference_npu_error"] = ref_npu_error

    runtime = runtime_stats = ref_runtime = ref_runtime_stats = None
    if pass_count != num_correct_trials:
        metadata["correctness_error"] = correctness_error
        return compiled_result(correctness=False, metadata=metadata)

    if measure_performance:
        timed = _time_both_models(
            torch=torch,
            new_model=new_model,
            ref_model=ref_model,
            ref_mode=ref_mode,
            get_inputs=get_inputs,
            process_input=process_input,
            seed=seed,
            num_warmup=num_warmup,
            num_perf_trials=num_perf_trials,
            torch_device=torch_device,
            l2_clear_size=l2_clear_size,
            metadata=metadata,
            last_new_out=last_new_out,
            ref_globals=ref_globals,
            memory_bandwidth_gbps=memory_bandwidth_gbps,
            peak_tflops=peak_tflops,
            excessive_speedup=excessive_speedup,
        )
        runtime, runtime_stats, ref_runtime, ref_runtime_stats = timed
        recheck = _post_timing_recheck(
            torch=torch,
            new_model=new_model,
            ref_model=ref_model,
            ref_mode=ref_mode,
            run_ref_cpu=run_ref_cpu,
            get_inputs=get_inputs,
            process_input=process_input,
            outputs_ok=outputs_ok,
            seed=seed,
            device=device,
            metadata=metadata,
        )
        if recheck is not None:
            return compiled_result(
                correctness=False,
                metadata=metadata,
                runtime=runtime,
                runtime_stats=runtime_stats,
                ref_runtime=ref_runtime,
                ref_runtime_stats=ref_runtime_stats,
            )

    return compiled_result(
        correctness=True,
        metadata=metadata,
        runtime=runtime,
        runtime_stats=runtime_stats,
        ref_runtime=ref_runtime,
        ref_runtime_stats=ref_runtime_stats,
    )


def _time_both_models(
    *,
    torch: Any,
    new_model: Any,
    ref_model: Any,
    ref_mode: str,
    get_inputs: Any,
    process_input: Any,
    seed: int,
    num_warmup: int,
    num_perf_trials: int,
    torch_device: Any,
    l2_clear_size: int,
    metadata: dict[str, Any],
    last_new_out: Any,
    ref_globals: dict[str, object],
    memory_bandwidth_gbps: float,
    peak_tflops: float,
    excessive_speedup: float,
) -> tuple[Any, Any, Any, Any]:
    """Time candidate and NPU reference; attach speedup and SOL metadata."""
    runtime = runtime_stats = ref_runtime = ref_runtime_stats = None

    def draw_inputs() -> list[Any]:
        return [process_input(item) for item in get_inputs()]

    seed_torch(seed)
    probe_inputs = draw_inputs()
    input_bytes = tensor_nbytes(probe_inputs)
    fresh_per_trial = input_bytes <= REFRESH_INPUT_BYTES_LIMIT
    metadata["timing_fresh_inputs"] = bool(fresh_per_trial)
    perf_box: list[Any] = [probe_inputs]
    prev_box: list[Any] = [None]

    def refresh_inputs() -> None:
        prev_box[0] = perf_box[0]
        perf_box[0] = draw_inputs()

    def timed(fn: Any) -> list[float]:
        seed_torch(seed)
        return time_execution_with_npu_event(
            lambda: fn(*perf_box[0]),
            [],
            num_warmup=num_warmup,
            num_trials=num_perf_trials,
            device=torch_device,
            setup=refresh_inputs if fresh_per_trial else None,
            l2_clear_size=l2_clear_size,
        )

    try:
        with torch.no_grad():
            runtime_stats = get_timing_stats(timed(new_model))
            runtime = runtime_stats["mean"]
            if ref_mode == "npu":
                ref_runtime_stats = get_timing_stats(timed(ref_model))
                ref_runtime = ref_runtime_stats["mean"]
                speedup = ref_runtime / runtime if runtime else 0.0
                metadata["speedup"] = float(f"{speedup:.4g}")
                metadata["excessive_speedup"] = bool(
                    speedup > excessive_speedup
                )
        attach_sol_metadata(
            metadata,
            kernel_ms=runtime if isinstance(runtime, int | float) else None,
            baseline_ms=(
                ref_runtime if isinstance(ref_runtime, int | float) else None
            ),
            bytes_moved=input_bytes + tensor_nbytes(last_new_out),
            bandwidth_gbps=float(memory_bandwidth_gbps),
            flops=task_declared_flops(ref_globals),
            peak_tflops=float(peak_tflops) if peak_tflops else None,
        )
    except Exception as exc:
        metadata["runtime_error"] = f"timing failed: {exc!r}"
    return runtime, runtime_stats, ref_runtime, ref_runtime_stats


def _post_timing_recheck(
    *,
    torch: Any,
    new_model: Any,
    ref_model: Any,
    ref_mode: str,
    run_ref_cpu: Any,
    get_inputs: Any,
    process_input: Any,
    outputs_ok: Any,
    seed: int,
    device: str,
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    """Fail the sample if a fresh-input re-check mismatches or raises."""
    try:
        seed_torch(seed + 1)
        recheck_raw = get_inputs()
        recheck_inputs = [process_input(item) for item in recheck_raw]
        torch.npu.synchronize(device=device)
        with torch.no_grad():
            if ref_mode == "npu":
                recheck_ref = ref_model(*recheck_inputs)
                torch.npu.synchronize(device=device)
            else:
                recheck_ref = run_ref_cpu(recheck_raw)
            recheck_new = new_model(*recheck_inputs)
            torch.npu.synchronize(device=device)
        if not outputs_ok(recheck_ref, recheck_new):
            metadata["correctness_error"] = (
                "post-timing fresh-input re-check failed: outputs are not "
                "a pure function of current inputs (caching or state drift)"
            )
            return metadata
    except Exception as exc:
        metadata["runtime_error"] = f"post-timing re-check failed: {exc!r}"
        return metadata
    return None
