conda activate vllm  

# 确保安装了 setproctitle

export CUDA_VISIBLE_DEVICES=0
python -c "import setproctitle; setproctitle.setproctitle('lablemodel'); import runpy; runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')" \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --max-model-len 32768 \
    --port 8000