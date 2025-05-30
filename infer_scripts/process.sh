# llama
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3_infer.yaml infer_scripts/infer_dataset.py training_configs/llama-3-8b-instruct-simpo.yaml --is_infer=True --report_to=none --output_dir=processed_datasets/llama-3-8b-instruct
# llama v2
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3_infer.yaml infer_scripts/infer_dataset.py training_configs/llama-3-8b-instruct-simpo-v2.yaml --is_infer=True --report_to=none --output_dir=processed_datasets/llama-3-8b-instruct-v2
# gemma
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3_infer.yaml infer_scripts/infer_dataset.py training_configs/gemma-2-9b-it-simpo.yaml --is_infer=True --report_to=none --output_dir=processed_datasets/gemma-2-9b-it
