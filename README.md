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

## Results and model improvements:
As you can see in the full code, starting from the forth epoch, I reached to 90% accuracy score. As the epochs are increasing the accuracy of the training score keeps rising, while the accuracy score on the validation set remains around 93% (not bad at all!), but I decided to stop here at 50 epochs to prevent overfitting.

Using the predict_classes() method I extracted the predicted labels and added them to the dataframe, to have a closer look at the results. The next thing I did to improve my model is to use a GloVe pre-trained Word Embadding, that been trained on milions of words.

The results that I received on the same number of epochs are very close, but as you can see in the code output, it didn't reach its max potential and it can be trained over more epochs and to reach a higher accuracy score.
The interesting thing to see is that even some of the tweets that were labeled wrong as hate tweets, has a pretty clear negative context.

## Conclusions:
My intent was training a model that will have the ability to analyze tweets and label them efficiently as hate and non-hate tweets. I decided to use Words Embedding and to train a Neural Networks model for this task. I explored 2 types of Words Embedding techniques which gave me a high accuracy score and seems to have great potential to even maximize it. 

Training this model on a much larger dataset and combining it with a larger pre-trained word embedding, can take this model to the next level, I also recommend to try and play with the model's hyperparameters and test the results (even consider performing grid search if you have the computing power to do so). 



## Credit & Code References:
1. https://www.kaggle.com/arkhoshghalb/twitter-sentiment-analysis-hatred-speech#train.csv
2. https://medium.com/r/?url=https%3A%2F%2Fnlp.stanford.edu%2Fprojects%2Fglove%2F
3. https://medium.com/greyatom/an-introduction-to-bag-of-words-in-nlp-ac967d43b428
4. https://medium.com/r/?url=https%3A%2F%2Fmachinelearningmastery.com%2Fuse-word-embedding-layers-deep-learning-keras%2F
5. https://medium.com/r/?url=https%3A%2F%2Fblog.keras.io%2Fusing-pre-trained-word-embeddings-in-a-keras-model.html
6. https://medium.com/@jonathan_hui/nlp-word-embedding-glove-5e7f523999f6



