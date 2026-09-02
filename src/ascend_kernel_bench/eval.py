"""Evaluation pipeline (README 4.5): static check -> build -> correctness ->
timing, each sample evaluated in an isolated worker subprocess.

The host entry (:func:`eval_sample`) writes a config JSON, spawns
``python -m ascend_kernel_bench.worker`` as a process-group leader via
Popen, kills the whole group on timeout, and reads the result from a JSON
file — CANN log spam on stdout never pollutes the result channel.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from .checker import check_custom_op_asc, check_model_new
from .config import EvalConfig, HardwareProfile
from .dataset import Task

__all__ = ["eval_sample", "eval_sample_on_device"]


def _fail(
    *,
    compiled: bool = False,
    compilation_error: str | None = None,
    runtime_error: str | None = None,
    **extra_metadata: object,
) -> dict:
    return {
        "compiled": compiled,
        "correctness": False,
        "runtime": None,
        "runtime_stats": None,
        "ref_runtime": None,
        "ref_runtime_stats": None,
        "metadata": {
            "compilation_error": compilation_error,
            "runtime_error": runtime_error,
            **extra_metadata,
        },
    }


def eval_sample(
    task: Task,
    sample_dir: Path,
    *,
    hardware: HardwareProfile,
    config: EvalConfig,
    device: str = "npu:0",
    measure_performance: bool = True,
) -> dict:
    """Host entry: static check, then isolated worker subprocess.

    ``sample_dir`` must already contain ``custom_op.asc`` and
    ``model_new.py``. The result dict is also written to
    ``sample_dir/eval_result.json``.
    """
    sample_dir = Path(sample_dir)
    model_new_path = sample_dir / "model_new.py"
    asc_path = sample_dir / "custom_op.asc"
    if not asc_path.is_file() or not model_new_path.is_file():
        return _fail(compilation_error="sample dir missing custom_op.asc or model_new.py")

    violations = check_custom_op_asc(asc_path.read_text(encoding="utf-8"))
    violations += check_model_new(model_new_path.read_text(encoding="utf-8"))
    if violations:
        result = _fail(
            compilation_error="; ".join(violations),
            static_check_error=violations,
        )
        _write_atomic(sample_dir / "eval_result.json", result)
        return result

    with tempfile.TemporaryDirectory(prefix="akb_eval_") as tmpdir:
        result_path = Path(tmpdir) / "result.json"
        cfg = {
            "task_py": task.task_py,
            "sample_dir": str(sample_dir),
            "cmake_arch": hardware.cmake_arch,
            "hardware_name": hardware.name,
            "device": device,
            "measure_performance": measure_performance,
            "seed": config.seed,
            "num_correct_trials": config.num_correct_trials,
            "num_perf_trials": config.num_perf_trials,
            "num_warmup": config.num_warmup,
            "precision": config.precision,
            "tolerances": config.tolerances,
            "excessive_speedup": config.excessive_speedup,
            "build_timeout": config.build_timeout,
            "result_path": str(result_path),
        }
        cfg_path = Path(tmpdir) / "cfg.json"
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
        env = dict(os.environ)
        env.setdefault("ASCEND_SLOG_PRINT_TO_STDOUT", "0")
        # The worker's build stage runs configure AND build, each allowed
        # build_timeout — the host budget must cover both plus the eval itself.
        timeout_s = config.eval_timeout + 2 * config.build_timeout
        proc = subprocess.Popen(
            [sys.executable, "-m", "ascend_kernel_bench.worker", str(cfg_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            env=env,
        )
        try:
            _, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            # start_new_session makes the worker a process-group leader, so
            # killpg reaps the worker and anything it forked (a hung NPU job
            # included) instead of leaking the group on the device.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            proc.wait()
            result = _fail(runtime_error=f"eval timed out after {timeout_s}s")
            _write_atomic(sample_dir / "eval_result.json", result)
            return result

        if proc.returncode != 0:
            err = (stderr or "").strip()
            result = _fail(
                runtime_error=err[-2000:] or f"worker exited with code {proc.returncode}"
            )
            _write_atomic(sample_dir / "eval_result.json", result)
            return result

        if not result_path.is_file():
            result = _fail(runtime_error="worker produced no result.json")
            _write_atomic(sample_dir / "eval_result.json", result)
            return result

        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            result = _fail(runtime_error=f"invalid worker JSON: {exc}")
            _write_atomic(sample_dir / "eval_result.json", result)
            return result

    _write_atomic(sample_dir / "eval_result.json", result)
    return result


def _write_atomic(path: Path, payload: dict) -> None:
    """Write JSON via a temp file + rename so interruptions never leave halves."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Worker body (runs inside the isolated subprocess)
# ---------------------------------------------------------------------------

_PRECISION_DTYPES = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}


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
    tolerances: dict,
    excessive_speedup: float,
    build_timeout: int,
) -> dict:
    """Build + correctness + timing in the current process (worker body)."""
    import torch
    import torch_npu  # noqa: F401  (registers the NPU backend)

    from .build import BuildError, build_custom_op
    from .timing import get_timing_stats, time_execution_with_npu_event

    sample_path = Path(sample_dir)
    asc_source = (sample_path / "custom_op.asc").read_text(encoding="utf-8")
    model_new_src = (sample_path / "model_new.py").read_text(encoding="utf-8")

    try:
        build_custom_op(
            asc_source, sample_path, cmake_arch=cmake_arch, timeout_s=build_timeout
        )
    except (BuildError, OSError) as exc:
        return _fail(compilation_error=str(exc))

    # Load task contract and generated wrapper.
    try:
        ref_globals: dict[str, object] = {}
        exec(compile(task_py, "<task.py>", "exec"), ref_globals)
        Model = ref_globals["Model"]
        get_init_inputs = ref_globals["get_init_inputs"]
        get_inputs = ref_globals["get_inputs"]
        task_tolerance = ref_globals.get("TOLERANCE")
        custom_check = ref_globals.get("custom_check")

        sys.path.insert(0, str(sample_path))
        try:
            custom_globals: dict[str, object] = {
                "__file__": str(sample_path / "model_new.py")
            }
            exec(
                compile(model_new_src, str(sample_path / "model_new.py"), "exec"),
                custom_globals,
            )
        finally:
            sys.path.remove(str(sample_path))
        ModelNew = custom_globals["ModelNew"]
    except Exception as exc:
        return _fail(compiled=True, runtime_error=f"module load failed: {exc!r}")

    torch_device = torch.device(device)
    device_index = int(device.split(":", 1)[1]) if ":" in device else 0
    torch.npu.set_device(device_index)

    dtype = getattr(torch, _PRECISION_DTYPES.get(precision, "float32"))
    if task_tolerance:
        atol = float(task_tolerance.get("atol", 1e-4))
        rtol = float(task_tolerance.get("rtol", 1e-4))
    else:
        tol = tolerances.get(precision, {"atol": 1e-4, "rtol": 1e-4})
        atol, rtol = float(tol["atol"]), float(tol["rtol"])

    def set_seed(value: int) -> None:
        torch.manual_seed(value)
        torch.npu.manual_seed(value)

    def process_input(x):
        if not isinstance(x, torch.Tensor):
            return x
        # Integer/bool inputs keep their dtype: masks, class ids, token ids.
        if torch.is_floating_point(x):
            return x.to(device=torch_device, dtype=dtype)
        return x.to(device=torch_device)

    def is_discrete(t) -> bool:
        return isinstance(t, torch.Tensor) and not (
            torch.is_floating_point(t) or torch.is_complex(t)
        )

    def tensor_match(ref, new) -> bool:
        """Shape plus dtype-aware value comparison.

        Floating/complex outputs use the configured tolerances; integer/bool
        outputs (indices, masks) require exact equality — torch.allclose
        rejects non-floating dtypes.
        """
        if not isinstance(new, torch.Tensor) or ref.shape != new.shape:
            return False
        if is_discrete(ref) and is_discrete(new):
            return bool(torch.equal(ref, new))
        try:
            return bool(torch.allclose(ref, new, atol=atol, rtol=rtol))
        except RuntimeError:
            return False

    def outputs_match(ref, new) -> bool:
        if isinstance(ref, torch.Tensor):
            return tensor_match(ref, new)
        if isinstance(ref, (tuple, list)):
            return (
                isinstance(new, (tuple, list))
                and len(ref) == len(new)
                and all(outputs_match(r, n) for r, n in zip(ref, new))
            )
        return bool(ref == new)

    set_seed(seed)
    init_inputs = get_init_inputs()
    try:
        # Identical RNG state for candidate and reference construction: tasks
        # with randomly-initialized parameters (convs, linears, norms) are only
        # winnable if ModelNew reproduces the reference's weights, which
        # requires constructing from the same seed (KernelBench convention).
        set_seed(seed)
        new_model = ModelNew(*init_inputs).to(device=torch_device, dtype=dtype)
        new_model.eval()
    except Exception as exc:
        return _fail(compiled=True, runtime_error=f"candidate model init failed: {exc!r}")

    # Reference protocol: torch_npu eager on NPU, same device, same inputs.
    # If the reference cannot run on NPU at all (torch_npu lacks the op),
    # fall back to a CPU reference: a correct candidate then effectively
    # extends torch_npu's operator coverage, which this benchmark wants to
    # surface rather than hide. CPU mode runs the reference in fp32.
    ref_mode = "npu"
    ref_model = None
    ref_model_cpu = None
    ref_npu_error: str | None = None
    try:
        set_seed(seed)
        ref_model = Model(*init_inputs).to(device=torch_device, dtype=dtype)
        ref_model.eval()
    except Exception as exc:
        ref_mode = "cpu"
        ref_npu_error = repr(exc)

    def run_ref_cpu(raw_inputs):
        nonlocal ref_model_cpu
        if ref_model_cpu is None:
            set_seed(seed)  # same construction RNG as the NPU reference path
            ref_model_cpu = Model(*init_inputs)
            ref_model_cpu.eval()
        cpu_inputs = [
            x.float() if isinstance(x, torch.Tensor) and torch.is_floating_point(x) else x
            for x in raw_inputs
        ]
        return ref_model_cpu(*cpu_inputs)

    # --- Correctness: seed chain, single reference per trial ---
    set_seed(seed)
    trial_seeds = [
        torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(num_correct_trials)
    ]
    pass_count = 0
    max_diff = 0.0
    correctness_error = ""
    with torch.no_grad():
        for trial, trial_seed in enumerate(trial_seeds):
            set_seed(trial_seed)
            raw_inputs = get_inputs()
            inputs = [process_input(x) for x in raw_inputs]
            set_seed(trial_seed)  # reset after get_inputs consumed RNG
            # torch_npu input generators (e.g. torch.rand) are not guaranteed to
            # be stream-ordered with raw <<<>>> kernel launches; sync so the
            # candidate never reads half-written inputs.
            torch.npu.synchronize(device=device)

            if ref_mode == "npu":
                try:
                    ref_out = ref_model(*inputs)
                    torch.npu.synchronize(device=device)
                except Exception as exc:
                    # Any NPU-reference failure falls back to the CPU reference,
                    # at any trial: a trial-order-dependent policy would score
                    # the same candidate differently depending on seed order.
                    ref_mode = "cpu"
                    ref_npu_error = f"trial {trial}: {exc!r}"
                    ref_out = run_ref_cpu(raw_inputs)
            else:
                ref_out = run_ref_cpu(raw_inputs)

            # Snapshot after the reference run: the pollution check must isolate
            # mutations made by the candidate, not by the reference itself.
            ref_snapshot = [
                x.clone() if isinstance(x, torch.Tensor) else x for x in inputs
            ]
            try:
                new_out = new_model(*inputs)
                torch.npu.synchronize(device=device)
            except Exception as exc:
                return _fail(
                    compiled=True,
                    correctness=False,
                    runtime_error=f"trial {trial}: candidate runtime error: {exc!r}",
                )

            # Input pollution check: candidate must not mutate its inputs.
            polluted = any(
                isinstance(a, torch.Tensor)
                and isinstance(b, torch.Tensor)
                and not torch.equal(a, b)
                for a, b in zip(inputs, ref_snapshot)
            )
            if polluted:
                return _fail(
                    compiled=True,
                    runtime_error=f"trial {trial}: candidate mutated its inputs",
                )

            # In CPU-reference mode compare on CPU (candidate output moved back).
            if ref_mode == "cpu":
                def _to_cpu(o):
                    if isinstance(o, torch.Tensor):
                        # fp32 CPU reference for floats; integer/bool outputs
                        # keep their dtype so indices and masks compare exactly.
                        return o.float().cpu() if torch.is_floating_point(o) else o.cpu()
                    return o
                cmp_ref = (
                    [_to_cpu(o) for o in ref_out]
                    if isinstance(ref_out, (tuple, list))
                    else _to_cpu(ref_out)
                )
                cmp_new = (
                    [_to_cpu(o) for o in new_out]
                    if isinstance(new_out, (tuple, list))
                    else _to_cpu(new_out)
                )
            else:
                cmp_ref, cmp_new = ref_out, new_out

            if custom_check is not None:
                ok = bool(custom_check(cmp_ref, cmp_new))
            else:
                ok = outputs_match(cmp_ref, cmp_new)
            if ok:
                pass_count += 1
            else:
                if (
                    isinstance(cmp_ref, torch.Tensor)
                    and isinstance(cmp_new, torch.Tensor)
                    and cmp_ref.shape == cmp_new.shape
                ):
                    try:
                        trial_max = torch.abs(cmp_ref - cmp_new).max().item()
                    except (RuntimeError, TypeError):
                        trial_max = float("nan")
                    if trial_max == trial_max:  # not NaN
                        max_diff = max(max_diff, float(trial_max))
                correctness_error = (
                    f"trial {trial}: output mismatch "
                    f"(passed {pass_count}/{num_correct_trials} so far)"
                )

    correctness = pass_count == num_correct_trials
    metadata: dict[str, object] = {
        "hardware": hardware_name,
        "precision": precision,
        "reference": ref_mode,
        "max_difference": max_diff,
        "correctness_trials": num_correct_trials,
        "correctness_passed": pass_count,
        "atol": atol,
        "rtol": rtol,
    }
    if ref_npu_error:
        metadata["reference_npu_error"] = ref_npu_error
    if not correctness:
        metadata["correctness_error"] = correctness_error
        return {
            "compiled": True,
            "correctness": False,
            "runtime": None,
            "runtime_stats": None,
            "ref_runtime": None,
            "ref_runtime_stats": None,
            "metadata": metadata,
        }

    # --- Performance (README 3.6): NPU Event, L2 clear per trial ---
    runtime = runtime_stats = ref_runtime = ref_runtime_stats = None
    if measure_performance:
        # Fresh inputs per trial defeat result-caching cheats, but regenerating
        # gigabyte-scale inputs a hundred times would dominate the measurement
        # (KernelBench L1 already ships a 6.4 GB input). Adaptive protocol:
        # input sets up to 256 MB are redrawn every trial; larger ones use a
        # fixed set (KernelBench's own behavior) and rely on the post-timing
        # fresh-input re-check below plus the excessive-speedup flag.
        REFRESH_INPUT_BYTES_LIMIT = 256 * 1024 * 1024

        def draw_inputs():
            return [process_input(x) for x in get_inputs()]

        set_seed(seed)
        probe_inputs = draw_inputs()
        input_bytes = sum(
            x.numel() * x.element_size()
            for x in probe_inputs
            if isinstance(x, torch.Tensor)
        )
        fresh_per_trial = input_bytes <= REFRESH_INPUT_BYTES_LIMIT
        metadata["timing_fresh_inputs"] = bool(fresh_per_trial)

        perf_box: list = [probe_inputs]
        prev_box: list = [None]

        def refresh_inputs():
            # Keep the previous set alive so the caching allocator cannot hand
            # back identical data_ptrs for a data_ptr-keyed cache to hit.
            prev_box[0] = perf_box[0]
            perf_box[0] = draw_inputs()

        def timed(fn):
            set_seed(seed)  # candidate and reference see identical input sequences
            return time_execution_with_npu_event(
                lambda: fn(*perf_box[0]),
                [],
                num_warmup=num_warmup,
                num_trials=num_perf_trials,
                device=torch_device,
                setup=refresh_inputs if fresh_per_trial else None,
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
                    metadata["excessive_speedup"] = bool(speedup > excessive_speedup)
            # CPU-reference mode has no NPU baseline: the candidate's absolute
            # runtime is recorded, but no speedup is computed.
        except Exception as exc:
            metadata["runtime_error"] = f"timing failed: {exc!r}"

        # Post-timing correctness re-check on one more FRESH input set (a seed
        # never used in the correctness trials): a candidate whose timed calls
        # replayed cached results — or whose state drifted across the ~100
        # timed calls — is caught here rather than by the timing protocol.
        try:
            set_seed(seed + 1)
            recheck_raw = get_inputs()
            recheck_inputs = [process_input(x) for x in recheck_raw]
            torch.npu.synchronize(device=device)
            with torch.no_grad():
                if ref_mode == "npu":
                    recheck_ref = ref_model(*recheck_inputs)
                    torch.npu.synchronize(device=device)
                else:
                    recheck_ref = run_ref_cpu(recheck_raw)
                recheck_new = new_model(*recheck_inputs)
                torch.npu.synchronize(device=device)
            if ref_mode == "cpu":
                def _to_cpu2(o):
                    if isinstance(o, torch.Tensor):
                        return o.float().cpu() if torch.is_floating_point(o) else o.cpu()
                    return o
                cmp_ref = (
                    [_to_cpu2(o) for o in recheck_ref]
                    if isinstance(recheck_ref, (tuple, list))
                    else _to_cpu2(recheck_ref)
                )
                cmp_new = (
                    [_to_cpu2(o) for o in recheck_new]
                    if isinstance(recheck_new, (tuple, list))
                    else _to_cpu2(recheck_new)
                )
            else:
                cmp_ref, cmp_new = recheck_ref, recheck_new
            ok = (
                bool(custom_check(cmp_ref, cmp_new))
                if custom_check is not None
                else outputs_match(cmp_ref, cmp_new)
            )
            if not ok:
                metadata["correctness_error"] = (
                    "post-timing fresh-input re-check failed: outputs are not "
                    "a pure function of current inputs (caching or state drift)"
                )
                return {
                    "compiled": True,
                    "correctness": False,
                    "runtime": runtime,
                    "runtime_stats": runtime_stats,
                    "ref_runtime": ref_runtime,
                    "ref_runtime_stats": ref_runtime_stats,
                    "metadata": metadata,
                }
        except Exception as exc:
            metadata["runtime_error"] = f"post-timing re-check failed: {exc!r}"
            return {
                "compiled": True,
                "correctness": False,
                "runtime": runtime,
                "runtime_stats": runtime_stats,
                "ref_runtime": ref_runtime,
                "ref_runtime_stats": ref_runtime_stats,
                "metadata": metadata,
            }

    return {
        "compiled": True,
        "correctness": True,
        "runtime": runtime,
        "runtime_stats": runtime_stats,
        "ref_runtime": ref_runtime,
        "ref_runtime_stats": ref_runtime_stats,
        "metadata": metadata,
    }
