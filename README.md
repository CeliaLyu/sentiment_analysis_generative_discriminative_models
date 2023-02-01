### Sentiment Analysis With Generative and Discriminative Models

**Xiaoxi Celia Lyu, Pingcheng Jian, Jingyang Zhang

Final project for ECE685D Introduction to Deep Learning, Duke University, Spring 2022. All authors contributed equally to this work.


# File description
## Datasets
- `IMDBDataset.csv`: The IMDB movie review data.
- `synthetic_data/`: The folder contains synthetic data generated with N-gram (`ngram.csv`) or Multinomial Naive Bayes (`bayes.csv`).

## Main scripts
### Generative models
- `ngram.ipynb`: The notebook for running N-gram experiments.
- `multinomial_naive_bayes.py`: The python script for running Multinomial Naive Bayes experiments.

### Discriminative models
- `lstm.ipynb`: The notebook for running LSTM experiments.
- `bert.ipynb`: The notebook for running BERT experiments.

## Utility files
- `data_utils.py`: Contains some functions for loading data.

# Required packages
- numpy, scikit-learn, nltk, wordcloud, textblob, beautifulsoup4, spacy
- torch, pytorch_pretrained_bert, tensorflow
- pandas, matplotlib, seaborn, tqdm
All these packages can be installed via `pip`.
