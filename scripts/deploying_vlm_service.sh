DEVICE=3

#CUDA_VISIBLE_DEVICES=$DEVICE python3 -m vllm.entrypoints.openai.api_server --model /ws/data/_model_weights/qwen/Qwen2-VL-7B-Instruct \
#  --served-model-name Qwen2-VL-7B-Instruct \
#  --tensor_parallel_size=1

CUDA_VISIBLE_DEVICES=$DEVICE python3 -m vllm.entrypoints.openai.api_server --model /ws/data/_model_weights/qwen/Qwen2-VL-2B-Instruct \
  --served-model-name Qwen2-VL-2B-Instruct \
  --tensor_parallel_size=1