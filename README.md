# Analyzing tweets semantic using Word Embedding

Social media has changed the world and gave individuals the opportunity to be heard and have an impact on society. In the past few years we have been witnessing a growing number of movements, protests and phenomena that started in social media.
This opportunity has its downsides too. The social media can provide a platform for spreading hate, racism, sexism and has a real potential of hurting people. Freedom of speech is important, but sometimes, detecting and preventing this type of content from being published is a must.

Using Kaggle, I extracted this Tweets dataset:
https://www.kaggle.com/arkhoshghalb/twitter-sentiment-analysis-hatred-speech#train.csv

that are classified as hate tweets and regular tweets. In this case, hate tweets are considered as tweets with racist and sexist context. My intent is to train a model that will have the ability to analyze the tweet and label it efficiently.

You can read my full article here:
https://medium.com/@natialuk/analyzing-tweets-semantic-using-word-embedding-9463e6fbeadb

## In this project you will find a few files: 
1. Analyzing tweets' semantic using Word Embedding.html - the full code file in html format
2. Analyzing tweets' semantic using Word Embedding.ipynb - the full code file in Jupyter notebook format
3. twitter_sentiment_analysis_train.csv the dataset I imported from Kaggle

## Libraries to import:
1. pandas
2. numpy 
3. matplotlib.pyplot
4. nltk
5. re
6. from keras.models: Model, Sequential
7. from keras.preprocessing.text: Tokenizer, text_to_word_sequence
8. from keras.preprocessing.sequence: pad_sequences
9. from keras.layers: Dense, Flatten, Embedding
10 from keras.optimizers: Adam
11. multiprocessing
12. from sklearn.decomposition: PCA
13. from keras.utils: plot_model
14. from gensim.models: Word2Vec

Credit & Code References:
1. https://www.kaggle.com/arkhoshghalb/twitter-sentiment-analysis-hatred-speech#train.csv
2. https://medium.com/r/?url=https%3A%2F%2Fnlp.stanford.edu%2Fprojects%2Fglove%2F
3. https://medium.com/greyatom/an-introduction-to-bag-of-words-in-nlp-ac967d43b428
4. https://medium.com/r/?url=https%3A%2F%2Fmachinelearningmastery.com%2Fuse-word-embedding-layers-deep-learning-keras%2F
5. https://medium.com/r/?url=https%3A%2F%2Fblog.keras.io%2Fusing-pre-trained-word-embeddings-in-a-keras-model.html
6. https://medium.com/@jonathan_hui/nlp-word-embedding-glove-5e7f523999f6



