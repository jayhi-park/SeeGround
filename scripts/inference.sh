DEVICE=3

#CUDA_VISIBLE_DEVICES=$DEVICE python3 parse_query/generate_query_data_scanrefer.py --anno_file /ws/data/scannet/scanrefer/meta_data/ScanRefer_filtered_val.json \
#  --model_name Qwen2-VL-2B-Instruct
CUDA_VISIBLE_DEVICES=$DEVICE python3 inference/inference_scanrefer.py --model_name Qwen2-VL-2B-Instruct
