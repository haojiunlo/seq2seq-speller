# Spelling Correction with Seq2seq Denoising Transformer

## Usage

### Prepare dataset
- data format
    - `.csv` file with correct text column
   
### Training

```bash
./run_training.sh
```

### Monitoring

```
tensorboard --logdir logs/bart_base_pretrain
```
