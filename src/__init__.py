"""AscendKernelBench evaluation engine.

Host orchestration is eval.py, the NPU worker body is eval_device.py; the
rest covers dataset, prompt, generation, checks, timing, and scoring.
"""
