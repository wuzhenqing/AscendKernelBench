# 本地推理服务部署指南

AscendKernelBench 的 LLM 接口为 OpenAI 兼容协议（README 3.9）。任何提供
OpenAI 兼容端点的服务都可使用，通过环境变量接入：

```bash
export OPENAI_BASE_URL=http://<host>:<port>/v1
export OPENAI_API_KEY=<your-key>
```

## 关键约束：NPU 物理分卡

**部署推理服务占用的 NPU 与评测使用的 NPU 必须物理分离，不得同卡。**
推理服务常驻显存会与评测进程争抢 HBM，导致计时失真甚至 OOM。

- 推理服务示例（8 卡机的 4-7 号卡）：
  `ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 vllm serve ...`
- 评测进程示例（0 号卡）：
  `ASCEND_RT_VISIBLE_DEVICES=0 python scripts/evaluate.py ...`

## vLLM-Ascend 部署示例

参考 [vLLM-Ascend](https://github.com/vllm-project/vllm-ascend) 官方文档安装后：

```bash
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
python -m vllm.entrypoints.openai.api_server \
    --model <model-path> --tensor-parallel-size 4 --port 8000
```

然后 `OPENAI_BASE_URL=http://127.0.0.1:8000/v1`。

## 远程服务

直接使用远程 OpenAI 兼容服务（如本项目测试用的 deepseek-v4-flash）时，
评测机无需为推理预留 NPU，但仍建议记录所用模型版本于
`runs/{run_name}/generation_config.yaml`（框架自动完成）。
