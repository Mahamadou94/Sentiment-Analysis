from pathlib import Path
import pytz
import os
import pandas as pd
from argparse import ArgumentParser
from pp_classifier.utilities import extract_annotations


def parse_arguments():
    parser = ArgumentParser(
        description="Preprocess raw multi-label dataset from LabelStudio into dataset containing "
        "input texts and labels"
    )
    parser.add_argument(
        'input_file',type= Path, help="Location of raw LabelStudio dataset"
    )
    parser.add_argument(
        'output_file', type=Path,
        help="Location of the output dataset after preprocessing",
    )
    args = parser.parse_args()
 
    assert args.input_file.exists(), f"{args.input_file} does not exist"
    return args

def main():
    args = parse_arguments()
    dataset = pd.read_json(args.input_file)
    dataset["text"] = dataset["data"].apply(lambda x: x["text"])
    dataset["labels"] = dataset["annotations"].apply(extract_annotations)
    dataset = dataset[["text", "labels"]]
    dataset = dataset.rename(columns={"text": "input_text"})
    dataset = dataset.drop_duplicates("input_text")
    dataset.to_json(args.output_file, orient="records", lines=True)

if __name__ == "__main__":
    main()
