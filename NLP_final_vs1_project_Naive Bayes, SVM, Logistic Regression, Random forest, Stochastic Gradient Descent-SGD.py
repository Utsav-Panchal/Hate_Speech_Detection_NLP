#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import textblob
import nltk
import time

import enchant #spelling
import itertools

#import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import re 
from textblob import TextBlob, Word, Blobber
#from spellchecker import SpellChecker
from nltk.tokenize import sent_tokenize, word_tokenize, TweetTokenizer
from nltk.stem import WordNetLemmatizer 

nltk.download("stopwords")
from nltk.corpus import stopwords

from sklearn import linear_model,svm
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

from collections import Counter


nltk.download('punkt')
from sklearn.feature_extraction.text import TfidfTransformer 
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 
nltk.download('wordnet')


# In[5]:


data = pd.read_csv("nlp_final_data.csv")
a=[0,5]
data=data.iloc[:,a]


# In[6]:


# data


# In[7]:


sns.set(font_scale=1)
sns.catplot("target", data=data, kind="count", palette="ch:start=.2,rot=-.3", height=5)
data['target'].value_counts()


# # Cleaning data

# In[8]:


data['tidy'] = data['text'].str.replace("[^A-Za-z']+", " ")


# In[9]:


d = enchant.Dict("en_US")
def arrangeSentence(sentence):
    sentence=sentence.lower()
    sentence=sentence.split()
    for i in range(len(sentence)):
        if d.check(''.join(''.join(s)[:2] for _, s in itertools.groupby(sentence[i]))):
            sentence[i]=''.join(''.join(s)[:2] for _, s in itertools.groupby(sentence[i]))
        else:
            sentence[i]=''.join(''.join(s)[:1] for _, s in itertools.groupby(sentence[i]))
    sentence=' '.join(sentence)
    return sentence


# In[10]:


stopwordlist = set(stopwords.words("english"))
to_remove = ['doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
             'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 
             'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", "don't","you",
            "your", "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
             'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves']
stopwordlist = set(stopwords.words('english')).difference(to_remove)


# In[12]:


lemmatizer = WordNetLemmatizer()
for i in range(8799):
    current=data.iloc[i,2]
    current=str(TextBlob(current)).lower().replace("  "," ")
    #type(current)
    current = arrangeSentence(current)
    current = ' '.join([lemmatizer.lemmatize(w) for w in current.split()])
    print(current)
    sentence1=[word for word in current.split() if word not in stopwordlist]
    sentence2=[word for word in sentence1 if len(word)>2]
    current = " ".join(sentence2)
    data.loc[i,'tidy']=str(current)


# # Counting number of tokens

# In[13]:


#tokenize the tweet data
tt = TweetTokenizer()
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

ps = PorterStemmer()


# In[14]:


# return word_tokenize(text)
def get_ngrams(text, n ):
    n_grams = ngrams(word_tokenize(text), n)
    return [ ' '.join(grams) for grams in n_grams]

def stemming(words):
    stem_words = []
    for w in words:
        w = ps.stem(w)
        stem_words.append(w)
    
    return stem_words


# In[15]:


lemmatizer = WordNetLemmatizer()


# In[17]:


data['stemmed']=data['tidy'].apply(tt.tokenize)
data['stemmed'] = data['stemmed'].apply(stemming)


# In[18]:


# data


# In[19]:


words = Counter()
for idx in data.index:
    words.update(data.loc[idx, "stemmed"])

#words.most_common()


# In[20]:


nltk.download('stopwords')
stopwords=nltk.corpus.stopwords.words("english")
whitelist = ["n't", "not"]
for idx, stop_word in enumerate(stopwords):
    if stop_word not in whitelist:
        del words[stop_word]
words.most_common(900)


# In[21]:


def word_list(processed_data):
    #print(processed_data)
    min_occurrences=5 
    max_occurences=5000 
    stopwords=nltk.corpus.stopwords.words("english")
    whitelist = ["n't","not"]
    wordlist = []
    
    whitelist = whitelist if whitelist is None else whitelist
    #print(whitelist)
    
    
    words = Counter()
    for idx in processed_data.index:
        words.update(processed_data.loc[idx, "stemmed"])

    for idx, stop_word in enumerate(stopwords):
        if stop_word not in whitelist:
            del words[stop_word]
    #print(words)

    word_df = pd.DataFrame(data={"word": [k for k, v in words.most_common() if min_occurrences < v < max_occurences],
                                 "occurrences": [v for k, v in words.most_common() if min_occurrences < v < max_occurences]},
                           columns=["word", "occurrences"])
    #print(word_df)
    word_df.to_csv("wordlist.csv", index_label="idx")
    wordlist = [k for k, v in words.most_common() if min_occurrences < v < max_occurences]
    # wordlist


# In[22]:


word_list(data)


# In[23]:


words = pd.read_csv("wordlist.csv")


# In[24]:


# words


# # Dividing data into training and testing

# In[27]:


X_train, X_test, y_train, y_test  = train_test_split(
        data['tidy'], 
        data.iloc[:,0],
        train_size=0.80, 
        random_state=1234)


# In[28]:


# y_train


# # First Approach- Dividing data according to TF-IDF 

# In[29]:


def scikit_TFIDF(m,n,Total_clean_train,Total_clean_test):
    vectorizer = CountVectorizer(min_df=1,max_features=m,ngram_range=(1,n))
    analyze = vectorizer.build_analyzer
    X_train = vectorizer.fit_transform(Total_clean_train).toarray()
    X_test = vectorizer.transform(Total_clean_test).toarray()
    transformer = TfidfTransformer()
    tfidf_train=transformer.fit_transform(X_train).toarray()
    tfidf_test=transformer.transform(X_test).toarray()
    return tfidf_train,tfidf_test


# In[30]:


X_train,X_test=scikit_TFIDF(2000 ,3,X_train,X_test)


# In[31]:


# X_train.shape


# In[32]:


from time import process_time


# In[35]:


logisticReg = linear_model.LogisticRegression(multi_class='ovr')

train_start_time = process_time()
logisticReg.fit(X_train, y_train)
train_end_time = process_time()

print(f"Training time : {train_end_time-train_start_time}")

logisticReg_prediction = logisticReg.predict(X_test)
logisticReg_accuracy = accuracy_score(y_test, logisticReg_prediction)
print("Training accuracy Score    : ",logisticReg.score(X_train, y_train))
print("Testing accuracy Score : ",logisticReg_accuracy )
print(classification_report(logisticReg_prediction, y_test))


# # Model 1) LOGISTIC REGRESSION

# In[30]:


start = time.time()
logisticReg = linear_model.LogisticRegression(multi_class='ovr')
logisticReg.fit(X_train, y_train)
Y_predict=(logisticReg.predict(X_test))
print(classification_report((Y_predict),y_test))

end = time.time()
print(end - start, " ms")


# # model 2) SUPPORT VECTOR MACHINE

# In[36]:


start = time.time()
lin_clf = svm.LinearSVC()
lin_clf.fit(X_train, y_train)
Y_SVC_linear_predic=lin_clf.predict(X_test)
svm_accuracy = accuracy_score(y_test, Y_SVC_linear_predic)
print(classification_report(Y_SVC_linear_predic,y_test))
end = time.time()
print(end - start, " ms")


# # Model 3) Naive Bayes Model

# In[38]:


from sklearn.naive_bayes import MultinomialNB

naiveByes_clf = MultinomialNB()

train_start_time = process_time()
naiveByes_clf.fit(X_train, y_train)
train_end_time = process_time()
print(f"Training time : {train_end_time-train_start_time}")

NB_prediction = naiveByes_clf.predict(X_test)

NB_accuracy = accuracy_score(y_test, NB_prediction)
print("Training accuracy Score    : ", naiveByes_clf.score(X_train, y_train))
print("Testing accuracy Score : ", NB_accuracy )
print(classification_report(NB_prediction, y_test))


# # Model 4) Random forest 
# 

# In[39]:


start = time.time()
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred= model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred)
print(classification_report(y_pred,y_test))
end = time.time()
print(end - start, " ms")


# # Model 5) Stochastic Gradient Descent-SGD Classifier

# In[49]:


from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(loss = 'hinge', penalty = 'l2', random_state=0)

train_start_time = process_time()
sgd_clf.fit(X_train,y_train)
train_end_time = process_time()

print(f"Training time : {train_end_time-train_start_time}")

sgd_prediction = sgd_clf.predict(X_test)
sgd_accuracy = accuracy_score(y_test, sgd_prediction)
print("Training accuracy Score    : ",sgd_clf.score(X_train, y_train))
print("Testing accuracy Score : ",sgd_accuracy )
print(classification_report(sgd_prediction,y_test))


# In[52]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 
              'Stochastic Gradient Decent'],
    'Test accuracy': [svm_accuracy, logisticReg_accuracy, 
              rf_accuracy, NB_accuracy, 
              sgd_accuracy]})

models.sort_values(by='Test accuracy', ascending=False)


# # Second approach- Dividing the data according to Bag of words approach

# In[41]:


count_vectorizer = CountVectorizer(min_df=10,max_features=2000,ngram_range=(1,4))
X = count_vectorizer.fit_transform(data['tidy']).toarray() 
Y = data.iloc[:,0].values


# In[42]:


# count_vectorizer


# In[43]:


# X


# In[44]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= 0.2, random_state=0)


# # Bag of words 
# # Model 1) Logistic regression
# 

# In[45]:


start = time.time()
logisticReg = linear_model.LogisticRegression(multi_class='ovr')
logisticReg.fit(X_train, y_train)
Y_predict=logisticReg.predict(X_test)
print(classification_report(Y_predict,y_test))
end = time.time()
print(end - start, " ms")


# # Model 2) SVM

# In[46]:


start = time.time()
lin_clf = svm.LinearSVC()
lin_clf.fit(X_train, y_train)
Y_SVC_linear_predic=lin_clf.predict(X_test)
print(classification_report(Y_SVC_linear_predic,y_test))
end = time.time()
print(end - start, " ms")


# # Bag of words Naive Bayes

# In[47]:


start = time.time()
gnb = GaussianNB()
Y_pred_Bayes = gnb.fit(X_train, y_train).predict(X_test)
print(classification_report(Y_pred_Bayes,y_test))
end = time.time()
print(end - start, " ms")


# # Model 4) Random forest

# In[48]:


start = time.time()
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred= model.predict(X_test)
print(classification_report(y_pred,y_test))
end = time.time()
print(end - start, " ms")


# In[ ]:




