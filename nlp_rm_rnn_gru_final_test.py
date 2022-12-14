# -*- coding: utf-8 -*-
"""NLP_Rm_RNN_GRU_FINAL_test.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PEzHWTrZj4HLi1TzMfYwuPhqLuCcK-Me
"""

import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
import tensorflow as tf
import keras
from keras.layers import Bidirectional, LSTM, GRU
from keras.models import Sequential
from keras.preprocessing import text
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


import nltk
import string
from unidecode import unidecode
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize, TweetTokenizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


from keras import Sequential, Model
from tensorflow.keras.optimizers import Adam,SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Layer, Input, Embedding, Conv1D, Bidirectional, LSTM, Dense, Dropout, BatchNormalization, GlobalMaxPooling1D, Flatten

import tensorflow.keras as keras
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.layers.experimental.preprocessing import Resizing

df = pd.read_csv("nlp_final_data.csv", encoding="ISO-8859-1")
a=[0,5]
df=df.iloc[:,a]
df = df.sample(n = 10000)

df.head()

def preprocess(text):
    # Convert to lowercase
    text = text.lower()
    # Remove mentions (@username), all URLs, all hashtags.  
    text = re.sub("@\S+|http\S+|#\S+", "", text)
        #Removing <"text"> type of text 
    text = re.sub('<.*?>+','',text)
    
    #Removing punctuations
    text = re.sub("[%s]" % re.escape(string.punctuation),'',text)
    
    #Removing new lines
    text = re.sub("\n",'',text)
    
    #Removing alphanumeric numbers 
    text = re.sub('\w*\d\w*','',text)
    # Tokenize
    tokenized_text = word_tokenize(text)
    cleaned_tokens = [t for t in tokenized_text if t.isalnum()]
    cleaned_string = " ".join(cleaned_tokens)
    return cleaned_string

def cleanDF(df):
    df["text"] = df["text"].apply(lambda x : preprocess(x))
    return df

import nltk
nltk.download('punkt')

df = cleanDF(df)
# df

lengths = df["text"].apply(lambda x : len(x.split(" "))) # mean + 3*sigma
lengths.hist()
plt.show()

mean = lengths.mean()
std = lengths.std()
approx_nseq = mean + 3*std
print(f"Mean: {mean:.4f}\nStd: {std:.4f}\nMean+3*Std = {approx_nseq:.4f}")

N_SEQ = 256

# Split into training and test sets
df_train, df_test = train_test_split(df, train_size = 0.7, shuffle = True, random_state = 42)

# df_test

# from wordcloud import WordCloud
#
# df_pos = df.loc[df["target"] == 0]
# df_neg = df.loc[df["target"] == 1]
#
# wc_pos = WordCloud(
#     width = 1600,
#     height = 800,
# ).generate(" ".join(df_pos["text"]))
#
# wc_neg = WordCloud(
#     width = 1600,
#     height = 800,
# ).generate(" ".join(df_neg["text"]))
#
# plt.figure(figsize = (20,20))
# plt.imshow(wc_pos)
# plt.title("Positive comments")
# plt.axis("off")
# plt.show()
#
# plt.figure(figsize = (20,20))
# plt.imshow(wc_neg)
# plt.title("Negative comments")
# plt.axis("off")
# plt.show()

t = Tokenizer()
t.fit_on_texts(df_train["text"])
word_index = t.word_index
N_vocab = len(t.word_index) + 1 # This +1 is used later 
print(N_vocab)

x_train = pad_sequences(t.texts_to_sequences(df_train["text"]), maxlen = N_SEQ)
x_test   = pad_sequences(t.texts_to_sequences(df_test["text"]), maxlen = N_SEQ)
y_train = df_train["target"].to_numpy(dtype = float).reshape(-1, 1)
y_test   = df_test["target"].to_numpy(dtype = float).reshape(-1, 1)

# # Maps each word in the embeddings vocabulary to it's embedded representation
# embeddings_index = {}
# with open("glove.twitter.27B.200d.txt/glove.twitter.27B.200d.txt", "r", errors="ignore", encoding="utf8") as f:
#     for line in f:
#         values = line.split()
#         word = values[0]
#         coefs = np.asarray(values[1:], dtype="float32")
#         embeddings_index[word] = coefs
#         print(1)
#

embeddings_index = dict()
f = open(r"glove.twitter.27B.200d.txt/glove.twitter.27B.200d.txt",encoding="utf8")
for l in f:
    lines = l.split()
    word = lines[0]
    cofi = np.asarray(lines[1:], dtype='float32')
    embeddings_index[word] = cofi
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

# Maps each word in our vocab to it's embedded representation, if the word is present in the GloVe embeddings
N_EMB = 200
embedding_matrix = np.zeros((N_vocab, N_EMB))
n_match = 0
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        n_match += 1
        embedding_matrix[i] = embedding_vector
print(n_match)

# Dimensionality of the hidden state h_t outputted by the GRU
DIM_HIDDEN = 64

model=Sequential()
#model.add(Input(shape=(N_SEQ,)))
model.add(Embedding(N_vocab,N_EMB, weights = [embedding_matrix],input_length = N_SEQ, trainable = False ))
model.add(GRU(units=DIM_HIDDEN, dropout = 0.2, return_sequences = True))
model.add(Flatten())
#model.add(Input(shape = (N_SEQ*DIM_HIDDEN,)))
model.add(Dense(1024, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(32, activation = "relu"))
model.add(Dense(1, activation = "sigmoid"))

check3 = tf.keras.callbacks.ModelCheckpoint(filepath="weights3.h5",monitor="val_accuracy",mode="max",save_best_only=True,)
log3 = CSVLogger('log3.csv', append=True, separator=',')

model.compile(optimizer = Adam(learning_rate = 0.001), loss = "binary_crossentropy", metrics = ["accuracy"])
print(model.summary())

history = model.fit(
    x_train, 
    y_train, 
    batch_size = 256, 
    epochs = 10,
    verbose=1,
    validation_data = (x_test, y_test), 
    callbacks=[check3,log3]
)

transfer_model = tf.keras.models.load_model('weights3.h5')

transfer_model.evaluate(x_test, y_test,batch_size=256, verbose=1)

N_EPOCHS=10
metrics = history.history
t_acc = metrics["accuracy"]
t_loss = metrics["loss"]
v_acc = metrics["val_accuracy"]
v_loss = metrics["val_loss"]

epochs = range(1, N_EPOCHS + 1)

plt.plot(epochs, t_acc)
plt.plot(epochs, v_acc)
plt.title("Accuracy")


plt.show()

y_pred = []
for i in x_test:
    i = np.expand_dims(i, 0)
    y_pred.append(1 if model.predict(i) > 0.5 else 0)

from sklearn.metrics import classification_report
print(classification_report( y_pred,y_test))

