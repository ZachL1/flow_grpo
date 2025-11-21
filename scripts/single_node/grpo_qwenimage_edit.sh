# 8 GPU
# conda install -c nvidia cuda-compiler
# pip install git+https://github.com/huggingface/diffusers.git
# pip install peft==0.17.0
# pip install deepspeed==0.17.2 accelerate==1.9.0 transformers==4.54.0
torchrun --standalone --nproc_per_node=8 --master_port=19501 scripts/train_qwenimage_edit.py --config config/grpo_ir.py:ir_qwenimage_edit_8gpu