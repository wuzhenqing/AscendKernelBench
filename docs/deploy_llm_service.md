# Connect an LLM service

AscendKernelBench generates code through an OpenAI-compatible Chat Completions endpoint. You can use a hosted provider or a separately deployed inference server. The benchmark does not start or manage an inference service.

Generation only reads task source and writes candidate files. It can run on macOS with the project's Python dependencies and access to your endpoint. Compiling and evaluating the generated Ascend C code requires a configured Linux Ascend host.

## Configure the connection

Set the endpoint's API base URL and credential in the shell that runs generation. Replace the example values with those supplied by your service:

```bash
export OPENAI_BASE_URL='https://your-service.example/v1'
export OPENAI_API_KEY='replace-with-your-service-key'
```

Use the **API base URL**, not a full `/chat/completions` URL. Some services use a different base path; follow that service's documentation. A server without authentication may still require a nonempty placeholder key for the client; use the value prescribed by its operator. Keep real credentials out of configuration files committed to Git.

Choose a model identifier exposed by your endpoint:

```bash
python scripts/generate.py \
  --task level1/19_ReLU \
  --model your-served-model \
  --hardware ascend910b2 \
  --n-samples 1 \
  --run-name endpoint_smoke
```

This command makes real API requests and saves one candidate. It does not compile or evaluate the candidate. A successful response checks the connection and basic response structure; it does not establish kernel correctness.

`--model` overrides `generation.model` in the evaluation YAML. The checked-in default is `deepseek-v4-flash`; this is a configuration value, not a guarantee that your service offers the model. There are no provider-specific adapters or `--base-url` / `--api-key` CLI options.

## Response format and retries

The client first requests a structured response with two string fields:

| Field | Required content |
| --- | --- |
| `custom_op_asc` | The complete `custom_op.asc` source, including the kernel, host launcher, and a process-local `TORCH_LIBRARY` / `TORCH_LIBRARY_IMPL` binding. |
| `model_new_py` | The complete `model_new.py` source defining `ModelNew`. |

If the structured request or parsing fails, the client makes a plain Chat Completions request and extracts fenced code blocks. It recognizes filename tags (`custom_op.asc`, `model_new.py`), then language tags (`cpp` / `asc`, `python`), and finally the first two blocks in Ascend C then Python order. A raw JSON answer to this fallback request is not decoded as JSON.

Both paths strip outer Markdown fences and check for these minimum markers:

- Ascend C: `__global__`, `__vector__`, `TORCH_LIBRARY`, and `TORCH_LIBRARY_IMPL`.
- Python: `class ModelNew` and `torch.ops.custom_op`.

These are content checks. The more detailed static checks run during evaluation, before compilation. See [task and candidate contracts](task_authoring.md).

The generation loop permits one retry after an invalid result or exception. Each attempt may include both a structured and a plain request, and the SDK may perform additional transport retries. One requested sample therefore does not necessarily equal one API request. The plain fallback is attempted after any structured-path exception, including connection and authentication errors, so inspect the underlying error when troubleshooting.

The client sends `temperature` and `max_tokens`; the selected model and endpoint must accept those parameters. The CLI exposes `--temperature`, while `generation.max_tokens` is set in YAML. The client constructor's default request timeout is 600 seconds; it has no dedicated CLI option and is independent of the NPU evaluation timeouts.

## Saved generation artifacts

For the example above, a successful sample is saved under:

```text
runs/endpoint_smoke/
├── generation_config.yaml
└── level1/19_ReLU/sample_0/
    ├── prompt.txt
    ├── custom_op.asc
    ├── model_new.py
    └── response_raw.txt
```

`response_raw.txt` is written when the returned response text is nonempty. `prompt.txt` contains the assembled benchmark prompt; the client adds its system message and structured-response instruction when submitting the request.

`generation_config.yaml` records the requested model identifier, sampling settings, hardware profile, and tasks. It does not capture the endpoint, immutable model revision, server configuration, token usage, or request IDs. Record relevant nonsecret service details alongside a published experiment if you need to reproduce it.

Use a new run name for a new experiment. Generation writes existing sample paths again, and creating a run replaces its generation configuration. Failed generation attempts print errors but do not create failure records. The final `saved/total` count is therefore part of checking that generation completed.

## Self-hosted inference and device isolation

For an Ascend-hosted service, use the installation and model-serving instructions for your chosen release in the [official vLLM Ascend documentation](https://docs.vllm.ai/projects/ascend/en/latest/). Select a compatible serving stack independently of the benchmark's build environment. Once the service is available, connect through the same environment variables above.

**An active inference server and benchmark workers must use separate physical NPUs.** Shared memory, compute, and bandwidth can cause out-of-memory failures and invalidate timing. Separate Python processes alone do not provide device isolation.

For example, an operator might reserve physical devices 4–7 for inference and physical device 0 for benchmarking. Restrict device visibility before starting each service or worker, then select the appropriate device in that process's visible device set:

```bash
# Run on the Linux Ascend evaluation host after preparing its environment.
ASCEND_RT_VISIBLE_DEVICES=0 python scripts/evaluate.py \
  --run-name endpoint_smoke \
  --hardware ascend910b2 \
  --device npu:0
```

Confirm the physical allocation and visible device mapping on your host. AscendKernelBench inherits `ASCEND_RT_VISIBLE_DEVICES` but does not check whether another process uses the same card. The batch evaluator processes samples sequentially; it is not a multi-device scheduler. You can also generate all samples first, stop the inference service, and evaluate afterward on an otherwise idle device.

The `ascend950pr` profile is reserved and requires hardware and toolchain validation before use. Its presence in the configuration directory does not establish support for either model serving or kernel evaluation.

For connection errors, missing sample files, and evaluation failures, see [troubleshooting](guide/troubleshooting.md).
