#! /bin/bash

for i in {1..10}; do
    echo "Start LoRA, epoch: $i"
    python lora.py --epoch "$i" --model_id Llama-3.2-3B-Instruct-pruned-0.95
    
    echo "LoRA completed, start testing..."
    sleep 10
    python test_lora.py --model_id Llama-3.2-3B-Instruct-pruned-0.95-LoRA-epoch-"$i"
done

git add .
git commit -m "Testing LoRA Completed, Result upload"
git push