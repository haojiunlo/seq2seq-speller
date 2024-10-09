import os
import argparse
import pandas as pd
from noising.noiser import noise


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', required=True, type=str, help='input .csv contains correct texts')
    parser.add_argument('--text_col', required=True, type=str, help='column name of correct texts')
    parser.add_argument(
        '--typo_probs', default=0.95, type=str, help='probabilty to generate typo for each word in input text')
    parser.add_argument(
        '--output_csv', default="data/synthetic_typo_data.csv", type=str, help='output .csv contains correct and synthetic typo texts')
    return parser.parse_args()


def main():
    df_input = pd.read_csv(args.input_csv)
    df_input["inputs"] = noise(df_input[args.text_col].tolist(), typo_probs=args.typo_probs, disable_tqdm=False)
    df_input.to_csv(args.output_csv, index=False)


if __name__ == '__main__':
    args = parse_args()
    main()
