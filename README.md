# NLP Projects: Regression, Text Generation, and BERT

This repository contains a series of Jupyter notebooks for different NLP projects, focusing on text regression, text generation using transformers, and fine-tuning a BERT model.

## Table of Contents

1. [Text Regression](#text-regression)
2. [Transformer (Text Generation)](#transformer-text-generation)
3. [BERT Model](#bert-model)
5. [Usage](#usage)
6. [Data Sources](#data-sources)
7. [Credits](#credits)

## Text Regression

### Description
This project involves scraping text data from Arabic websites on a specific topic, preprocessing the text, and training various RNN-based models to predict a relevance score for each text.

### Steps
1. **Data Collection**: Using Scrapy or BeautifulSoup to scrape text data from Arabic websites.
2. **Dataset Preparation**: Creating a dataset with two columns: `Text` (Arabic text) and `Score` (relevance score between 0 to 10).
3. **NLP Pipeline**: Preprocessing the text data (tokenization, stemming, lemmatization, removing stop words, discretization).
4. **Model Training**: Training RNN, Bidirectional RNN, GRU, and LSTM models with hyperparameter tuning.
5. **Evaluation**: Evaluating models using standard metrics and BLEU score.

## Transformer (Text Generation)

### Description
This project focuses on fine-tuning a pre-trained GPT-2 model using a custom dataset and generating new text based on a given sentence.

### Steps
1. **Installation**: Installing `pytorch-transformers`.
2. **Model Loading**: Loading the pre-trained GPT-2 model.
3. **Fine-Tuning**: Fine-tuning the GPT-2 model on a custom dataset.
4. **Text Generation**: Generating new paragraphs based on a given input sentence.

### Tutorial
Follow the tutorial [here](https://gist.github.com/mf1024/3df214d2f17f3dcc56450ddf0d5a4cd7) for detailed steps on fine-tuning GPT-2.

## BERT Model

### Description
This project involves using the pre-trained `bert-base-uncased` model for text classification tasks using a dataset from Amazon reviews.

### Steps
1. **Data Preparation**: Downloading and preparing the dataset from [Amazon Reviews](https://nijianmo.github.io/amazon/index.html).
2. **Model Setup**: Setting up the BERT embedding layer.
3. **Fine-Tuning**: Fine-tuning the BERT model with appropriate hyperparameters.
4. **Evaluation**: Evaluating the model using metrics like Accuracy, Loss, F1 score, BLEU score, and BERT-specific metrics.
5. **Conclusion**: Summarizing the performance and insights from using the pre-trained BERT model.

## Data Sources

- **Text Regression**: Scraped from various Arabic websites using Scrapy or BeautifulSoup.
- **Text Generation**: [Dataset](https://www.kaggle.com/datasets/shailajakodag1/netflix-titlescsv/data)
- **BERT Model**: [Amazon Reviews Dataset](https://nijianmo.github.io/amazon/index.html).

## Credits

- Project supervised by Pr. Elaachak Lotfi at Université Abdelmalek Essaadi, Faculté des Sciences et Techniques de Tanger, Département Génie Informatique.
- Inspired by various tutorials and open-source projects in the NLP community.
