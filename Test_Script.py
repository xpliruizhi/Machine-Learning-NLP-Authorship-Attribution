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

def Read_from_TestSet_Dict():
    # Function to read line from files
    # Put this file in the folder of "train_tweets.zip"
    # Return a tuple list [('8746',['I','am','@'...]),(),(),(),...] 
    
    #pdb.set_trace()
    #my_docs = []
    my_docs_key = defaultdict(list)
    print("begin to read data files,time:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))
    myzip = zipfile.ZipFile('train_tweets.zip')
    ziplist = myzip.namelist()
    ## loop each txt file
    fileobj = myzip.open(ziplist[0])
    j = 0
    for line in fileobj:
        # must use utf-8 to store different languages
        # remove "/n" at each line 
        myline = line.decode('utf-8').strip()
        # use first 1 TAB to cut the string
        # j = j+1
        # if j ==18:
            # pdb.set_trace()
        line_list = myline.split('\t',1)
        #print(line_list)
        value_txt = BM25_PREPROCESSING_REMOVE_STOPWORD(line_list[1])
        #value_txt = line_list[1]
        # my_docs.append((line_list[0],value_txt))
        if line_list[0] in my_docs_key.keys():
            #pdb.set_trace()
            my_docs_key[line_list[0]].append(value_txt)
        else:
            my_docs_key[line_list[0]] = []
            my_docs_key[line_list[0]].append(value_txt)
    # break
    myzip.close()
    #pdb.set_trace()
    print("File Dict Reading Finish,time:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))
    return my_docs_key

def get_test_dataset(Doc_Dict,Author_List):
    test_docs = []
    pdb.set_trace()
    for Author in Author_List:
        doc_author = Doc_Dict[Author]
        for singletweet in doc_author:
            test_docs.append((Author,singletweet))
    return test_docs
    
def doc(text):
    return text

wordnet_lemmatizer = WordNetLemmatizer()

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
    
def Processing_for_second_time(Docs_Potential,potential_author,real_author,predict_sentence):
    pdb.set_trace()
    vectorizer = TfidfVectorizer(stop_words="english",
                             lowercase=False,
                             tokenizer= doc,
                             min_df = 10,
                             preprocessor=regular_expression,
                             #strip_accents =None,
                             token_pattern = r'\S+',
                             ngram_range=(1, 3),
                             max_features = 120000)
    training_features = vectorizer.fit_transform(X_train)
    classifier = OneVsRestClassifier(LinearSVC())
    print("start fitting,time:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))
    classifier.fit(training_features, Y_train)
    #print("done fitting")
    print("done fitting:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))
    #df= pd.DataFrame({'test_text':test_data})
    #test_real_features = vectorizer.transform(df['test_text'])
    test_real_features = vectorizer.transform(predict_sentence)
    Y_pred_output = classifier.predict(test_real_features)
    return (real_author,Y_pred_output)


Author_List = ['2080', '9846', '8432', '240', '9198', '7391', '3235', '9269', '3147']
Doc_Dict = Read_from_TestSet_Dict()
predict_sentence = ['http', '://', 'twitpic', '.', 'com', '/', 'fso8h', '-', '"', 'flame', '"', '#', 'cheap', '+', 'trick']
Docs_Potential = get_test_dataset(Doc_Dict,Author_List)
real_author = '3235'
#print(Docs_Potential)
#print(len(Docs_Potential))
pdb.set_trace()
Processing_for_second_time(Docs_Potential,Author_List,real_author,predict_sentence)