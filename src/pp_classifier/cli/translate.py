from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
from easynmt import EasyNMT


def parse_arguments():
    parser = ArgumentParser(
        description="Translate the preprocess documents into other languages"
    )
    parser.add_argument(
        "input_file", type=Path, help="Location of preprocessed multi-label dataset"
    )
    parser.add_argument(
        "output_file", type=Path, help="Location of translated multi-label dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="opus-mt",
        choices=["opus-mt", "mbart50-m2m", "m2m_100_418M"],
        help="Translation model choices"
    )
    parser.add_argument(
        "--language_targets",
        type=str,
        nargs="+",
        default=["de", "fr"],
        help="The list of language targets",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for the translation model",
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU"
    )

    parser.add_argument(
        "--language_source",
        type=str,
        default="en",
        help="The language source text"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    # Load up the model
    model = EasyNMT(args.model, max_loaded_models=3, device="cuda" if args.gpu else "cpu")

    # Load up the dataset
    dataset = pd.read_json(args.input_file, lines=True, encoding='utf-8')

    # Translate the text
    texts = dataset["input_text"].tolist()
    translated_texts = []
    for lang in args.language_targets:
        translations = model.translate(
            texts,
            target_lang=lang,
            batch_size=args.batch_size,
            source_lang=args.language_source,
            document_language_detection=False,
            show_progress_bar=True,
        )
        translations_df = pd.DataFrame(
            {"input_text": translations, "labels": dataset["labels"]}
        )
        translated_texts.append(translations_df)

    # Combine the original and translated texts
    combined_dataset = pd.concat([dataset] + translated_texts, ignore_index=True)
    combined_dataset.to_json(args.output_file, orient="records", lines=True)


if __name__ == "__main__":
    main()
