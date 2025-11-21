# 1 GPU
# pip install paddlepaddle-gpu==2.6.2
# pip install paddleocr==2.9.1
# pip install python-Levenshtein
# accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port 29501 scripts/train_sd3.py --config config/grpo.py:general_ocr_sd3_1gpu

# export WANDB_MODE=offline
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port 29501 scripts/train_flux_kontext.py --config config/grpo_ir.py:ir_flux_kontext_1gpu

# 9 GPU
# accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=8 --main_process_port 29501 scripts/train_flux_kontext.py --config config/grpo_ir.py:ir_flux_kontext
