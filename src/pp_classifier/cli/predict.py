from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import sparknlp
from sparknlp.base import *


def parse_arguments():
    parser = ArgumentParser(description="Perform inference on topic")
    parser.add_argument(
        "input_file", type=Path, help="Location of input file need to be predicted"
    )
    parser.add_argument(
        "output_file", type=Path, help="Location of output file after being predicted"
    )
    parser.add_argument("model_location", type=Path, help="Location of spark-nlp model")
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    spark = sparknlp.start(gpu=False)
    # Load model
    pipeline_model = PipelineModel.load(str(args.model_location))

    # Load data
    data = pd.read_json(str(args.input_file),lines=True, encoding='utf-8')
    data = spark.createDataFrame(data)
    data.cache()

    # Predict the label
    result = pipeline_model.transform(data)

    # Store result
    result.repartition(1).write.format("json").save(str(args.output_file))


if __name__ == "__main__":
    main()
