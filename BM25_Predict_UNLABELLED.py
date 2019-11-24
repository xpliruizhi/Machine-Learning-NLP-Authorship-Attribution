#!/usr/bin/env python
# coding: utf-8

import pdb
import re
import zipfile
import datetime
from nltk.tokenize import word_tokenize, wordpunct_tokenize
from nltk.corpus import stopwords
#from collections import Counter,defaultdict
import numpy as np
import gc
from collections import Counter,defaultdict
from gensim import summarization
import multiprocessing as mp
from multiprocessing import Pool
import math
import csv
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
from nltk.stem import WordNetLemmatizer

ISOTIMEFORMAT = '%H:%M:%S'
stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()

def BM25_PREPROCESSING(Sentence):
    #Function to Tokecnize the sentence into word, and also split punctuation
    #Remain the Capital words     
    New_line = []
    words = wordpunct_tokenize(Sentence)     
    return words

def BM25_PREPROCESSING_REMOVE_STOPWORD(Sentence):
    #Function to Tokecnize the sentence into word, and also split punctuation
    #Remain the Capital words 
    New_line = []
    words = wordpunct_tokenize(Sentence)
    for word in words:
        if word.lower() in stop_words:
            continue
        else:
            New_line.append(word.lower())
    return New_line

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

def Read_from_TestSet():
    # Function to read line from files
    # Put this file in the folder of "train_tweets.zip"
    # Return a tuple list [('8746',['I','am','@'...]),(),(),(),...] 
    #pdb.set_trace()
    my_docs = []
    #my_docs_key = defaultdict(list)
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
        j = j+1
        #pdb.set_trace()
        line_list = myline.split('\t',1)
        #print(line_list)
        value_txt = BM25_PREPROCESSING_REMOVE_STOPWORD(line_list[1])
        my_docs.append((line_list[0],value_txt))
    myzip.close()
    #pdb.set_trace()
    print("File Reading Finish,time:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))
    return my_docs

def Read_from_test_set_unlabelled():
    # Read unlabelled set to predict
    my_docs_unlabelled = []
    i = 0
    with open('test_tweets_unlabeled.txt', 'rb') as files:
        for line in files:
            myline = line.decode('utf-8').strip()
            value_txt = BM25_PREPROCESSING_REMOVE_STOPWORD(myline)
            my_docs_unlabelled.append(value_txt)
    #pdb.set_trace()
    files.close()
    print("length of test set:"+str(len(my_docs_unlabelled)))
    return my_docs_unlabelled
    
def average_idf_cal(idf_dic):
    total_len = len(idf_dic.keys())
    return sum(idf_dic.values())/total_len

def build_inverted_index_BM25(new_bm):
    inverted_index = defaultdict(Counter)
    ave_idf = average_idf_cal(new_bm.idf)
    for i in range(0,new_bm.corpus_size-1):
        single_doc = list(new_bm.f[i].keys())
        for word in single_doc:
            inverted_index[word][i] = new_bm.get_score([word],i,ave_idf)
    return inverted_index

def doc_by_BM25_diff_title(query_list,invert_index_dict, document_keys):
    #pdb.set_trace()
    word_index_counter = Counter()
    for word in query_list:
        if word in invert_index_dict.keys():    
            word_index_counter = word_index_counter + invert_index_dict[word]
    #print(word_index_counter)
    doc_ranking = word_index_counter.most_common(10)
    #pdb.set_trace()
    rank_list = [document_keys[element[0]] for element in doc_ranking]
    #print(str(rank_list)+": "+str(query_list))
    return rank_list
    #return doc_ranking

def predict_BM25_Model_multipeprocess(invert_index_dict,train_set_labels,test_set,process_index):
    index_test_label = 0
    total_len = len(test_set)
    corrrect_label = 0
    result = []    
    for singleTweet in test_set:
        #pdb.set_trace()
        #query_list = BM25_PREPROCESSING(singleTweet)
        #query_list = singleTweet
        my_document_l = doc_by_BM25_diff_title(singleTweet,invert_index_dict,train_set_labels)
        if len(my_document_l) == 0:
            result.append((singleTweet, train_set_labels[0],set(train_set_labels[1:50])))
        else:
            result.append((singleTweet, my_document_l[0],set(my_document_l)))
        # if test_set_labels[0] == my_document_l[0]:
            # corrrect_label = corrrect_label + 1
        index_test_label = index_test_label + 1
        if index_test_label%100 == 0:
            print("processs"+str(process_index)+":"+str(index_test_label)+"data has processed.Total:"+str(total_len)+"Time:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))
    print("process" +str(process_index)+ "Finished")
    #print(corrrect_label/total_len)
    return (process_index,result)

def predict_BM25_Model_multipeprocess_2nd_process(invert_index_dict,train_set_labels,test_set,process_index,Doc_Dict):
    index_test_label = 0
    total_len = len(test_set)
    corrrect_label = 0
    result = []    
    for singleTweet in test_set:
        #pdb.set_trace()
        #query_list = BM25_PREPROCESSING(singleTweet)
        #query_list = singleTweet
        my_document_l = doc_by_BM25_diff_title(singleTweet,invert_index_dict,train_set_labels)
        #print(set(my_document_l),process_index)
        if len(my_document_l) == 0:
            Docs_Potential = get_test_dataset(Doc_Dict,list(set(train_set_labels))[:5])
            predicted_label = Processing_for_second_time(Docs_Potential,singleTweet)
            result.append((predicted_label, train_set_labels[0]))
            #print(predicted_label,my_document_l[0],set(my_document_l))
        elif len(set(my_document_l)) == 1:
            result.append((my_document_l[0],set(my_document_l)))
            #print(my_document_l[0],my_document_l[0],set(my_document_l))
        else:
            Docs_Potential = get_test_dataset(Doc_Dict,list(set(my_document_l)))
            predicted_label = Processing_for_second_time(Docs_Potential,singleTweet)
            result.append((predicted_label,set(my_document_l)))
            #print(predicted_label,my_document_l[0],set(my_document_l))
        # if test_set_labels[0] == my_document_l[0]:
            # corrrect_label = corrrect_label + 1
        index_test_label = index_test_label + 1
        if index_test_label%100 == 0:
            print("processs"+str(process_index)+":"+str(index_test_label)+"data has processed.Total:"+str(total_len)+"Time:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))
    print("process" +str(process_index)+ "Finished")
    #print(corrrect_label/total_len)
    return (process_index,result)

def chunks(arr, m):
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]

def multiple_processing_prediction():
    if __name__=='__main__':
        test_set,invert_index_dict,train_set_docs,train_set_labels = Run_preprocess_for_multiple()
        #Doc_Dict = Read_from_TestSet_Dict()
        chunklist = chunks(test_set, 11)
        print('Start Doing Multiple Processing now')
        Doc_Dict = Read_from_TestSet_Dict()
        process_number = len(chunklist)
        res = []
        p = Pool(process_number)
        for i in range(process_number):
            #pdb.set_trace()
            res.append(p.apply_async(predict_BM25_Model_multipeprocess_2nd_process,args=(invert_index_dict,train_set_labels,chunklist[i],i,Doc_Dict,)))
            print(str(i) + "process started")
        p.close()
        p.join()
        #pdb.set_trace()
        res = [item.get() for item in res]
        res.sort()
        simple = []
        for it in res:
            #print(it[0])
            simple.extend(it[1])
        i = 0
        j = 0
        #pdb.set_trace()
        result_list = []
        #Doc_Dict = Read_from_TestSet_Dict()
        for single in simple:
            # j = j+1
            # if j%1000 ==0:
                # print(str(j)+"Recordings Predict Finished. Time:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))
            i = i+1
            # result_list.append((i,single[1]))
            result_list.append((i,single[0]))
        #pdb.set_trace() 
        print("Write Final result to CSV File")
        headers = ["Id","Predicted"]
        with open('my_sml_results.csv','w',newline='') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(headers)
            f_csv.writerows(result_list)
        f.close()
        print("Write result Finished")

def Run_preprocess_for_multiple():
    train_set = Read_from_TestSet()
    #test_set,train_set = create_set_test_train(my_docs)
    test_set = Read_from_test_set_unlabelled()
    print('Read Predict and trainning set success! can do the following step now')
    #pdb.set_trace()
    train_set_transpose_order = [single for single in zip(*train_set)]
    train_set_docs = list(train_set_transpose_order[1])
    train_set_labels = list(train_set_transpose_order[0])
    train_set_transpose_order = None
    train_set = None
    gc.collect()
    print(len(train_set_docs))
    #pdb.set_trace()
    print("BM25 initialise....TIME:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))
    my_bm25 = summarization.bm25.BM25(train_set_docs)
    print("BM25 Initialization Finished....TIME:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))

    my_docs_detail = None
    del my_docs_detail

    print("Building Inverted index...TIME:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))
    invert_index_dict = build_inverted_index_BM25(my_bm25)
    print("Inverted index finished...TIME:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))
    my_bm25 = None
    del my_bm25
    gc.collect()
    return test_set,invert_index_dict,train_set_docs,train_set_labels

def get_test_dataset(Doc_Dict,Author_List):
    test_docs = []
    for Author in Author_List:
        doc_author = Doc_Dict[Author]
        for singletweet in doc_author:
            test_docs.append((Author,singletweet))
    return test_docs

def doc(text):
    return text

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
    
def Processing_for_second_time(train_docs, predict_sentence):
    #train_docs = get_test_dataset(my_dict,potential_author)
    
    train_data = np.array(train_docs)
    df = pd.DataFrame(train_data, columns=['author', 'text'])
    author = df['author']
    text = df['text']
    vectorizer = TfidfVectorizer(
    #stop_words="english",
                             lowercase=False,
                             tokenizer= doc,
                             min_df = 10,
                             preprocessor=regular_expression,
                             #strip_accents =None,
                             token_pattern = r'\S+',
                             ngram_range=(1, 3),
                             max_features = 120000)
    #pdb.set_trace()
    training_features = vectorizer.fit_transform(text)
    classifier = OneVsRestClassifier(LinearSVC())
    #print("start fitting,time:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))
    classifier.fit(training_features, author)
    #print("done fitting")
    #print("done fitting:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))
    #df= pd.DataFrame({'test_text':test_data})
    #test_real_features = vectorizer.transform(df['test_text'])
    #pdb.set_trace()
    test_real_features = vectorizer.transform([predict_sentence])
    Y_pred_output = classifier.predict(test_real_features)
    return Y_pred_output[0]

multiple_processing_prediction()



#pdb.set_trace()
#print("Success")

#print(invert_index_dict['Name'])

#test_set_transpose_order = [single for single in zip(*test_set)]
# test_set_docs = list(test_set_transpose_order[1])
# test_set_labels = list(test_set_transpose_order[0])
# test_set_transpose_order = None
# test_set = None
# gc.collect()
# print(len(test_set_docs))

#cross_validation_BM25_Model(train_set_labels,invert_index_dict,test_set_docs,test_set_labels)

# def BM25_PREPROCESSING_REMOVE_STOPWORD_URL(Sentence):
    # #Function to Tokecnize the sentence into word, and also split punctuation
    # #Remain the Capital words 
    # New_line = []
    
    # p1 = r"http[s]?\:\//[0-9A-za-z.]*\/[0-9A-za-z.]*"
    # pattern1 = re.compile(p1)
    # p2 = r"http[s]?\:\//([0-9A-za-z.]*)\/[0-9A-za-z.]*"
    # pattern2 = re.compile(p2)
    # key = pattern1.findall(Sentence)
    # key2 = pattern2.findall(Sentence)
    # #print(key)
    # #print(key2)
    # if len(key)>0 and len(key2)>0:
        # for key_all, key_simple in zip(key,key2):
            # #print(key_all)
            # #print(key_simple)
            # Sentence = Sentence.replace(key_all,key_simple)
    # words = wordpunct_tokenize(Sentence)
    # for word in words:
        # if (word.lower()in stop_words or word=='.' or word==','):
            # continue
        # else:
            # New_line.append(word.lower())
    # return New_line

#def cross_validation_BM25_Model_multipeprocess(invert_index_dict,train_set_labels,test_set,process_index):
    # index_test_label = 0
    # total_len = len(test_set)
    # corrrect_label = 0
    # result = []    
    # for singleTweet in test_set:
        # #pdb.set_trace()
        # #query_list = BM25_PREPROCESSING(singleTweet)
        # #query_list = singleTweet
        # my_document_l = doc_by_BM25_diff_title(singleTweet[1],invert_index_dict,train_set_labels)
        # if len(my_document_l) == 0:
            # result.append((singleTweet[0],singleTweet[1], 'no_result'))
        # else:
            # result.append((singleTweet[0],singleTweet[1], my_document_l[0]))
        # # if test_set_labels[0] == my_document_l[0]:
            # # corrrect_label = corrrect_label + 1
        # index_test_label = index_test_label + 1
        # if index_test_label%100 == 0:
            # print("processs"+str(process_index)+":"+str(index_test_label)+"data has processed.Total:"+str(total_len)+"Time:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))
    # print("process" +str(process_index)+ "Finished")
    # #print(corrrect_label/total_len)
    # return result
    
# def create_set_test_train(my_docs_key):
    # # random create test set and train set by numpy 
    # # total: 328932
    # # set the test set length to change the length of both sets 
    # # present length is 10%
    # test_set_length = 33000
    # test_set = []
    # train_set = []
    # np.random.seed(12345)
    # np.random.shuffle(my_docs_key)
    # test_set = my_docs_key[0:test_set_length]
    # train_set = my_docs_key[test_set_length:]
    # return test_set,train_set 
    
#def cross_validation_BM25_Model(doc_labels,invert_index_dict,test_set_docs,test_set_labels):
    # index_test_label = 0
    # total_len = len(test_set_labels)
    # corrrect_label = 0 
    # for singleTweet in test_set_docs:
        # #pdb.set_trace()
        # #query_list = BM25_PREPROCESSING(singleTweet)
        # #query_list = singleTweet
        # my_document_l = doc_by_BM25_diff_title(singleTweet,invert_index_dict, doc_labels)
        # if test_set_labels[index_test_label] == my_document_l[0]:
            # corrrect_label = corrrect_label + 1
        # index_test_label = index_test_label + 1
        # if index_test_label%100 == 0:
            # print(str(index_test_label)+"data has processed.Total:"+str(total_len)+"Time:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))
    # print("accuracy is:")
    # print(corrrect_label/total_len)