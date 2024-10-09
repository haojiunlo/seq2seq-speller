import os
import argparse
import glob
import logging
import tempfile

from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from zipfile import ZipFile

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        required=True,
        type=str,
        help="path contains parameters, logging, metrics and checkpoints",
    )
    parser.add_argument(
        "--export_path",
        required=True,
        type=str,
        help="path to export model",
    )
    return parser.parse_args()


def main():
    logging.info("Loading model and exporting to onnx")
    saved_model_path = os.path.join(args.save_path, "checkpoints")
    model = ORTModelForSeq2SeqLM.from_pretrained(
        saved_model_path, from_transformers=True
    )
    tokenizer = AutoTokenizer.from_pretrained(saved_model_path)

    with tempfile.TemporaryDirectory() as tmpdirname:
        logging.info("saving exported onnx model")
        model.save_pretrained(tmpdirname)
        tokenizer.save_pretrained(tmpdirname)

        logging.info(
            f"zipping the onnx model file: {glob.glob(os.path.join(tmpdirname, '*'))}"
        )

        os.makedirs(args.export_path, exist_ok=True)
        with ZipFile(os.path.join(args.export_path, "onnx.zip"), "w") as zip:
            # writing each file one by one
            for f in glob.glob(os.path.join(tmpdirname, "*")):
                zip.write(f)


if __name__ == "__main__":
    args = parse_args()
    main()
