export PATH=/datadrive1/ken/miniconda3/envs/videollama/bin/:$PATH

while true; do
    # Generate a random port number between 1024 and 65535
    PORT=$((RANDOM % (65535 - 1024 + 1) + 1024))
    # Check if the port is available using 'nc'
    nc -z 127.0.0.1 $PORT >/dev/null 2>&1
    # If the port is available, break the loop
    if [[ $? -ne 0 ]]; then
        break
    fi
done
echo $PORT

expdir=exp/finetune_llama2_7b
mkdir -p $expdir

# Need to replace model_path and data_path, current data_path is just for testing

python finetune.py \
    --model_path /datadrive1/ken/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852/ \
    --data_path /datadrive1/brian/braingpt_finetuning/data/dataset/testdata/ \
    --batch_size 1 \
    --chunk_size 2048 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 3 \
    --num_warmup_steps 0.03 \
    --weight_decay 0.001 \
    --lr_scheduler_type cosine \
    --outputdir $expdir \
    --logfile $expdir/log.txt \
    --log_interval 10 \
    --save_interval 100 \
    --lora_config config/lora_config.json \
    --master_port $PORT \
