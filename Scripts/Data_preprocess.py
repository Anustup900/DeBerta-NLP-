import os
import re
import sklearn
import nltk
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


train_list = ["../Dataset/Actualdata/Books/Bookstrain.txt","../Dataset/Actualdata/Dvd/Dvdtrain.txt","../Dataset/Actualdata/Electronics/Electronicstrain.txt","../Dataset/Actualdata/Kitchen/Kitchentrain.txt"]

test_list = ["../Dataset/Actualdata/Books/Bookstest.txt","../Dataset/Actualdata/Dvd/Dvdtest.txt","../Dataset/Actualdata/Electronics/Electronicstest.txt","../Dataset/Actualdata/Kitchen/Kitchentest.txt"]


stopword = stopwords.words('english') 

def preprocess(sentence):
	sentence = re.sub('[^\w\s]'," ",str(sentence))
	sentence = re.sub('[^a-zA-Z]'," ",str(sentence))
	sents = word_tokenize(sentence)
	new_sents = " "
	for i in range(len(sents)):
		if(sents[i].lower() not in stopword):
			new_sents+=sents[i].lower()+" "
	return new_sents

def preprocess_test(choice):

	the_file = test_list[choice]
	#print(the_file)
	with open(the_file,'r',encoding='utf-8') as f:
		test_data = f.readlines()

	corpus_test = []

	for i in range(400):
		sent = test_data[i]
		sent = sent[0:len(sent)-1]
		corpus_test.append(preprocess(sent))

	#print(corpus_test[0])

	label_test = np.zeros(400)
	label_test[0:200] = 1


	return [corpus_test,label_test]


def preprocess_train(choice):


	the_file = train_list[choice]
	#print(the_file)
	with open(the_file,'r',encoding='utf-8') as f:
		train_data = f.readlines()


	corpus_train = []

	for i in range(1600):
		sent = train_data[i]
		sent = sent[0:len(sent)-1]
		corpus_train.append(preprocess(sent))

	#print(corpus_train[0])

	label_train = np.zeros(1600)
	label_train[0:800] = 1

	return [corpus_train,label_train]

def preprocessing(train_choice,test_choice):

	corpus_train,label_train = preprocess_train(train_choice)

	corpus_test,label_test = preprocess_test(test_choice)

	return corpus_train,label_train,corpus_test,label_test
