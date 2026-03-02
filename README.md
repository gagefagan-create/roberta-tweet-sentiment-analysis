# RoBERTa Tweet Sentiment Analysis

Comparative sentiment and emotion analysis of tweets mentioning **Joe Biden** and **Donald Trump** during the 2020 U.S. presidential election cycle. This project applies multiple NLP techniques — from lexicon-based methods to transformer models — to classify emotional tone across ~1.16 million tweets.

## Project Overview

This research analyzes public sentiment on Twitter by comparing three classification approaches:

| Method | Type | Description |
|--------|------|-------------|
| **NRCLex** | Lexicon-based | Maps words to 10 emotion categories using the NRC Emotion Lexicon |
| **TextBlob** | Rule-based | Computes polarity scores to classify Positive / Negative / Neutral |
| **RoBERTa** | Transformer (neural) | Fine-tuned `cardiffnlp/twitter-roberta-base` on 100K political tweets for 4-class emotion detection |

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 01 | `01_data_preprocessing_and_eda.ipynb` | Text cleaning, exploratory analysis, word frequencies, n-grams, LDA topic modeling with pyLDAvis cluster visualization, TextBlob & VADER baselines |
| 02 | `02_biden_sentiment_analysis.ipynb` | Full NRCLex + TextBlob + fine-tuned RoBERTa pipeline on ~498K Biden tweets |
| 03 | `03_trump_sentiment_analysis.ipynb` | Full NRCLex + TextBlob + fine-tuned RoBERTa pipeline on ~661K Trump tweets |
| 04 | `04_roberta_finetuning.ipynb` | Fine-tunes `twitter-roberta-base` on 100K political tweets using VADER + NRCLex pseudo-labels for domain-specific emotion classification |

## Key Findings

- **RoBERTa** identified **joy** as the dominant emotion in Trump-related tweets (72%) and Biden-related tweets (49%), diverging from lexicon-based methods
- **NRCLex** found **surprise** dominant in Trump tweets and **positive** dominant in Biden tweets
- **TextBlob** classified the majority of tweets in both datasets as Neutral, highlighting limitations of rule-based polarity scoring
- The discrepancy between methods underscores how model architecture and training data influence sentiment classification

## Dataset

- **Source**: Twitter API (collected October 2020)
- **Trump tweets**: ~970K raw → ~661K after English language filtering
- **Biden tweets**: ~777K raw → ~498K after English language filtering

## Tech Stack

- Python 3.10+
- pandas, NumPy, matplotlib, seaborn
- NLTK (tokenization, stopwords, VADER)
- NRCLex (NRC Emotion Lexicon)
- TextBlob (rule-based sentiment)
- Hugging Face Transformers (fine-tuned `cardiffnlp/twitter-roberta-base`)
- gensim + pyLDAvis (topic modeling)
- Google Colab (GPU runtime for RoBERTa inference)

## Setup

1. Open any notebook in [Google Colab](https://colab.research.google.com/)
2. Upload your tweet CSV datasets to Google Drive
3. Update the `DATA_PATH` variable at the top of each notebook
4. Run all cells (GPU runtime recommended for notebook 02 & 03)

## Author

**Gage Fagan**
