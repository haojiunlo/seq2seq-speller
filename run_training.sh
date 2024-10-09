HF_DATASETS_CACHE=/data/cache \
python run_experiment.py \
    --model_name_or_path facebook/bart-base \
    --data au/train.csv \
    --val_data au/val.csv \
    --test_data au/test.csv \
    --dynamic_noise True \
    --max_steps 400000 \
    --batch_size 128 \
    --learning_rate 2e-4 \
    --gradient_accumulation_steps 8 \
    --experiment_name bart_base_pretrain \
    --report_to tensorboard \
    --fp16 True
