from test_set_read import Read_from_TestSet,Read_from_test_set_unlabelled
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import nltk
import re
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
#nltk.download('wordnet')
import datetime
import pdb
ISOTIMEFORMAT = '%H:%M:%S'

from nltk.stem import WordNetLemmatizer
train_data = Read_from_TestSet()
test_data = Read_from_test_set_unlabelled()
#df = pd.DataFrame(train_data, columns=['author', 'text'])
#pdb.set_trace()
#test_text = df_test['test_text']
train_data = np.array(train_data)
#author
df = pd.DataFrame(train_data, columns=['author', 'text'])
print(df.shape)
author = df['author']

text = df['text']

X_train, X_test, Y_train, Y_test = train_test_split(text, author,test_size=0.01,random_state=90051)
print("Training set has {} instances. Test set has {} instances.".format(X_train.shape[0], X_test.shape[0]))

def doc(text):
    return text
wordnet_lemmatizer = WordNetLemmatizer()
"""def lemmatize_text(text):
    lemmatized_output = ' '.join([wordnet_lemmatizer.lemmatize(w) for w in text])
    return lemmatized_output"""
def regular_expression(words):
    temp = []
    #for words in text:
    if isinstance(words,str):
        words = re.sub(r'[:|,|)|(|\|/]','',words)
        words = re.sub(r'[\'|"|]','',words)
        words = re.sub('!+','!',words)
        words = re.sub(r'\.+',r'.',words)
        words = re.sub(r'\$+',r'$',words)
        words = re.sub(r'\*+',r'*',words)
        words = words.replace("http","")
        words=words.strip()
        if not words.isupper():
            words = words.lower()
            #print("single words",words)
        return [wordnet_lemmatizer.lemmatize(words)] 
    else:
        for word in words:
            word = re.sub(r'[:|,|)|(|\|/]','',word)
            word = re.sub(r'[\'|"|]','',word)
            word = re.sub('!+','!',word)
            word = re.sub(r'\.+',r'.',word)
            word = re.sub(r'\$+',r'$',word)
            word = re.sub(r'\*+',r'*',word)
            word = word.replace("http","")
            word=word.strip()
            if not word.isupper():
                #print(word)
                word = word.lower()
            temp.append(word)
        words = temp
        
    lemmatized_output = [wordnet_lemmatizer.lemmatize(w) for w in words]
    lemmatized_output.append(str(len(lemmatized_output)))
    #lemmatized_output = ' '.join([wordnet_lemmatizer.lemmatize(w) for w in words])
    #print("lemma",lemmatized_output)
    return lemmatized_output

vectorizer = TfidfVectorizer(stop_words="english",
                              lowercase=False,
                             tokenizer= doc,
                             min_df = 10,
                             preprocessor=regular_expression,
                             #strip_accents =None,
                             token_pattern = r'\S+',
                             ngram_range=(1, 3),
			     max_features = 120000
                            )

training_features = vectorizer.fit_transform(X_train)    
test_features = vectorizer.transform(X_test)
list_feature_name = vectorizer.get_feature_names()

#print(training_features.shape)
#print(list_feature_name[:500])
#real_test_features = vectorizer.transform(test_text)
#classifier = OneVsRestClassifier(LogisticRegression(n_jobs=1, C=1))
classifier = OneVsRestClassifier(LinearSVC())
#classifier = LogisticRegression(n_jobs=1, C=1e5)
#mlb = MultiLabelBinarizer()
#Y_train1 = mlb.fit_transform(Y_train)
#classifier = OneVsRestClassifier(LogisticRegression(n_jobs=1, C=1e5))

print("start fitting,time:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))

classifier.fit(training_features, Y_train)
#print("done fitting")
print("done fitting:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))

#Y_test1 = mlb.transform(Y_test)
#print(Y_test1)
Y_test_pred = classifier.predict(test_features)
#print(Y_test_pred)
print(accuracy_score(Y_test, Y_test_pred))

df= pd.DataFrame({'test_text':test_data})
test_real_features = vectorizer.transform(df['test_text'])
Y_test_pred_output = classifier.predict(test_real_features)

df = pd.DataFrame(Y_test_pred_output,columns=['Predicted'])
df.index = df.index + 1
df.index.name = 'Id'
df.to_csv('predict_SML_Pro.csv')