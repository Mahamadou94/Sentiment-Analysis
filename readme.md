First# Introduction
Topic classification on news and reviews dataset

# Installation

1. Create a virtual python environment, could be from venv or conda
2. Install the project using `pip install -e .`
3. Run the installed command lines

# Examples
The installed command can be checked in `src/pp_classifier/cli` folder

Data could be obtained through exporting the labelled dataset from LabelStudio in JSON format

There are three steps:
1. `preprocess`: Extracting text and labels from LabelStudio dataset
2. `translate`: Adding translated text to the dataset
3. `train`: Train multilabel classification model

Any extra flags for each command can be listed through `--help` flag. For example `train --help`

Example of steps. Let's say we have exported the labelled data from LabelStudio and stored it in ./dataset/raw folder.


```
# Extract text and labels
preprocess ./dataset/raw/raw_data.json ./dataset/preprocessed/cleaned_data.json

# Translate the cleaned dataset
translate ./dataset/preprocessed/cleaned_data.json ./dataset/preprocessed/translated_data.json

# Train the model
train ./dataset/preprocessed/translated_data.json ./outputs/model
```

# Resources
1. The data should be sourced from LabelStudio in JSON format. The documentation can be found [here](https://labelstud.io/guide/)
2. The Translation library is done by making use of [EasyNMT](https://github.com/UKPLab/EasyNMT) library.
3. The multilabel classifier is built with [Spark NLP](https://nlp.johnsnowlabs.com/docs/en/quickstart) model.