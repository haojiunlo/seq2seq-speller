import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging
import argparse
import datasets
import datetime

import pandas as pd

from utils.datasets import Seq2SeqDataset, DynamicTypoDataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)


logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name_or_path', required=True, type=str, help='pretrained model name or path')
    parser.add_argument(
        '--config_name_or_path', default=None, type=str, help='config name or path')
    parser.add_argument(
        '--tokenizer_name_or_path', default=None, type=str, help='tokenizer name or path')
    parser.add_argument(
        '--from_scratch', default=False, type=bool, help='set to `True` wil not use pretrained weight and overwrite model_name_or_path')
    parser.add_argument(
        '--data', required=True, type=str, help='.csv data to be split into train and val set')
    parser.add_argument(
        '--val_data', default=None, type=str, help='.csv for valudation. if set `data` will not be split')
    parser.add_argument(
        '--test_data', default="data/golden_set.csv", type=str, help='.csv data for testing')
    parser.add_argument(
        '--input_col', default="inputs", type=str, help='column name of incorrect text')
    parser.add_argument(
        '--target_col', default="targets", type=str, help='column name of correct text')
    parser.add_argument(
        '--dynamic_noise', default=False, type=bool,
        help='whether to dynamically generate typo during training. If `True` the `input_col` will not be used')
    parser.add_argument(
        '--typo_probs', default=0.95, type=float,
        help='probabilty to generate typo for each word in input text. only used when `dynamic_noise` set to `True`')
    parser.add_argument(
        '--val_size', default=0.001, type=float, help='proportion of the dataset to be split into val set')
    parser.add_argument(
        '--min_length', default=3, type=int, help='model generation min length')
    parser.add_argument(
        '--max_length', default=48, type=int, help='model generation max length')
    parser.add_argument(
        '--num_beams',
        default=2,
        type=int,
        help="Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
        "which is used during ``evaluate`` and ``predict``.")
    parser.add_argument(
        '--num_epochs', default=5, type=int, help="num of train epochs")
    parser.add_argument(
        '--max_steps',
        default=-1,
        type=int,
        help="If set to a positive number, the total number of training steps to perform. Overrides `num_train_epochs`."
    )
    parser.add_argument(
        '--batch_size', default=128, type=int, help="num of batch size")
    parser.add_argument(
        '--learning_rate', default=5e-5, type=float, help="The initial learning rate for [`AdamW`] optimizer.")
    parser.add_argument(
        '--early_stopping', default=-1, type=int, help="If `early_stopping` greater than 0 enable early stopping with patience `early_stopping`")
    parser.add_argument(
        '--warmup_steps', default=500, type=int, help="Number of steps used for a linear warmup from 0 to `learning_rate`.")
    parser.add_argument(
        '--gradient_accumulation_steps', default=1, type=int, help="Number of updates steps to accumulate the gradients for, before performing a backward/update pass.")
    parser.add_argument(
        '--experiment_name', default="my_experiment", type=str, help="experiment name for saving checkpoints and logging")
    parser.add_argument(
        '--resume_from_checkpoint',
        default=None,
        type=str,
        help="If a str, local path to a saved checkpoint as saved by a previous instance of [Trainer]. "
        "If a bool and equals True, load the last checkpoint in *args.output_dir* as saved by a previous instance of [Trainer].")
    parser.add_argument(
        '--report_to',
        default='mlflow',
        type=str,
        help='The list of integrations to report the results and logs to. '
        'Supported platforms are `azure_ml`,`comet_ml`, `mlflow`, `tensorboard` and `wandb`.'
        'Use `all` to report to all integrations installed, `none` for no integrations.'
    )
    parser.add_argument(
        '--overwrite_output_dir', default=False, type=bool, help="If `True`, overwrite the content of the output directory.")
    parser.add_argument(
        '--fp16', default=True, type=bool, help="Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training."
    )
    return parser.parse_args()


def main():
    # load tokenizer
    logging.info("init tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path)

    # load dataset
    logging.info("loading datasets")
    df_trn = pd.read_csv(args.data).dropna()
    if args.val_data:
        df_val = pd.read_csv(args.val_data).dropna()
    else:
        df_trn, df_val = train_test_split(df_trn, test_size=args.val_size, random_state=1)

    df_test = pd.read_csv(args.test_data)
    df_test = df_test[~df_test[args.target_col].isna()].dropna(
        subset=[args.input_col]).reset_index(drop=True)

    logging.info(
        f"Train size: {len(df_trn)}, Val size: {len(df_val)}, Test size: {len(df_test)}")

    if args.dynamic_noise:
        datasets_trn = DynamicTypoDataset(df_trn[args.target_col].tolist(),
                                          tokenizer, typo_probs=args.typo_probs, max_length=args.max_length)
    else:
        datasets_trn = Seq2SeqDataset(df_trn[args.input_col].tolist(),
                                      df_trn[args.target_col].tolist(),
                                      tokenizer, max_length=args.max_length)

    datasets_val = Seq2SeqDataset(df_val[args.input_col].tolist(),
                                  df_val[args.target_col].tolist(),
                                  tokenizer, max_length=args.max_length)

    datasets_ts = Seq2SeqDataset(df_test[args.input_col].tolist(),
                                 df_test[args.target_col].tolist(),
                                 tokenizer, max_length=args.max_length)

    # init model
    logging.info("init model")
    if args.from_scratch:
        config = AutoConfig.from_pretrained(args.config_name_or_path)
        model = AutoModelForSeq2SeqLM.from_config(config)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    model.config.max_length = args.max_length
    model.config.min_length = args.min_length
    model.config.num_beams = args.num_beams

    model.resize_token_embeddings(len(tokenizer))

    # load rouge for validation
    speller_metrics = datasets.load_metric("utils/speller_metrics.py")

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions
        inputs = pred.inputs

        # all unnecessary tokens are removed
        inputs[inputs == -100] = tokenizer.pad_token_id
        input_str = tokenizer.batch_decode(inputs, skip_special_tokens=True)
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(
            labels_ids, skip_special_tokens=True)

        output = speller_metrics.compute(
            predictions=pred_str, references=label_str, inputs=input_str)

        return output
    

    training_args = Seq2SeqTrainingArguments(
        sortish_sampler=True,
        output_dir=f"checkpoints/{args.experiment_name}",
        max_steps=args.max_steps,
        num_train_epochs=args.num_epochs,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=0.1,
        predict_with_generate=True,
        logging_dir=f"logs/{args.experiment_name}",
        logging_steps=1000,
        logging_first_step=True,
        save_steps=2000,
        evaluation_strategy="steps",
        eval_steps=2000,
        save_total_limit=3,
        report_to=args.report_to,
        overwrite_output_dir=args.overwrite_output_dir,
        include_inputs_for_metrics=True,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        metric_for_best_model='eval_loss',
        load_best_model_at_end=True,
        ignore_data_skip=True
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=args.early_stopping)] if args.early_stopping > 0 else None

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=datasets_trn,
        eval_dataset=datasets_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks = callbacks
    )

    logging.info("start training")
    if isinstance(args.resume_from_checkpoint, str):
        trainer.train(True if args.resume_from_checkpoint.lower() == "true" else args.resume_from_checkpoint)
    else:
        trainer.train()

    test_metrics = trainer.evaluate(datasets_ts, metric_key_prefix="test")
    trainer.log_metrics("test", test_metrics)
    trainer.save_model()


if __name__ == '__main__':
    today = datetime.datetime.today().strftime('%Y%m%d')
    args = parse_args()
    main()
