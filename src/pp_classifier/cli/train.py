from pathlib import Path
from argparse import ArgumentParser
import os
import pandas as pd
import pyspark.sql
import sparknlp
from sparknlp.annotator import *
from sparknlp.base import *


def parse_arguments():
    parser = ArgumentParser(description="Train multi-label classification model")
    parser.add_argument("input_file", type=Path, help="Location of multi-label dataset")
    parser.add_argument("output_model", type=Path, help="Location of trained model")
    parser.add_argument(
        "--model",
        default="sent_small_bert_L4_768",
        type=str,
        help="Sentence embedding model, refer to: https://nlp.johnsnowlabs.com/models, search for `Bert Sentence"
             "embedding`",
    )
    parser.add_argument(
        "--gpu",
        default=False,
        action="store_true",
        help="Using gpu or not, default False",
    )
    args = parser.parse_args()
    return args


def create_pipeline(args):
    document_assembler = (
        DocumentAssembler().setInputCol("input_text").setOutputCol("document")
    )
    sentence_detector=(
        SentenceDetector()
        .setInputCols(["document"])
        .setOutputCol("sentence")
    )
    embedding_sentence = (
        UniversalSentenceEncoder.pretrained("tfhub_use_multi", "xx")
        .setInputCols(["sentence"])
        .setOutputCol("sentence_embeddings")   
    )
    classifier_dl = (
        MultiClassifierDLApproach()
        .setInputCols(["sentence_embeddings"])
        .setOutputCol("class")
        .setLabelColumn("labels")
        .setMaxEpochs(100)
        .setLr(3e-4)
        .setThreshold(0.5)
        .setValidationSplit(0.1)     
    )
    finisher = Finisher().setInputCols("class").setOutputCols("topics_output")
    pipeline = Pipeline(
        stages=[document_assembler,sentence_detector,embedding_sentence, classifier_dl,finisher]
    )
    return pipeline

def main():
    args = parse_arguments()
    spark = sparknlp.start(gpu=args.gpu)
    # Define pipeline
    pipeline = create_pipeline(args)
    # Load dataset
    raw_df = pd.read_json(args.input_file, lines = True, encoding ='utf-8-sig')
    train_dataset: pyspark.sql.DataFrame = spark.createDataFrame(raw_df)
    train_dataset.repartition(1000)
    # Train the pipeline
    pipeline_model = pipeline.fit(train_dataset)
    # Store model
    pipeline_model.write().overwrite().save(str(args.output_model))

if __name__ == "__main__":
    main()