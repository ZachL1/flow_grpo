# 1 GPU
# pip install paddlepaddle-gpu==2.6.2
# pip install paddleocr==2.9.1
# pip install python-Levenshtein
# accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port 29501 scripts/train_sd3.py --config config/grpo.py:general_ocr_sd3_1gpu

# 1 GPU
# pip install -e .
# conda install -c nvidia cuda-compiler
# pip install git+https://github.com/huggingface/diffusers.git
# pip install peft==0.17.0
# pip install deepspeed==0.17.2 accelerate==1.9.0 transformers==4.54.0 lpips

export HF_HOME="../.cache/huggingface"
# huggingface-cli login --token xxx

# pip install -U wandb
# wandb login --relogin xxx

# export WANDB_MODE=offline
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port 9001 scripts/train_flux_kontext.py --config config/grpo_ir.py:ir_flux_kontext_1gpu

# 8 GPU
# accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=8 --main_process_port 9001 scripts/train_flux_kontext.py --config config/grpo_ir.py:ir_flux_kontext
