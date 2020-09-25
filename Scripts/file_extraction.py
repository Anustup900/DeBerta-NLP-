import os
import re
import numpy as np 
import pandas as pd 


def extract_files(pos_file,neg_file,num):
	
	with open(pos_file,'r',encoding='utf-8') as f:
		pos = f.readlines()
	with open(neg_file,'r',encoding='utf-8') as f:
		neg = f.readlines()


	corpus_pos = []
	flag = False

	for i in range(len(pos)):
		if(pos[i]=="<review_text>\n"):
			flag = True
			new_str = " "
			continue
		elif(pos[i]=="</review_text>\n"):
			flag = False
			corpus_pos.append(new_str)
			continue
		if(flag):
			if(len(pos[i])>1):
				sent = pos[i]
				sent = sent[0:len(sent)-1]
				if(sent[0]=='\t'):
					sent = sent[1:len(sent)-1]
			new_str+=sent  
	

	corpus_neg = []
	flag = False


	for i in range(len(neg)):
		if(neg[i]=="<review_text>\n"):
			flag = True
			new_str = " "
			continue
		elif(neg[i]=="</review_text>\n"):
			flag = False
			corpus_neg.append(new_str)
			continue
		if(flag):
			if(len(neg[i])>1):
				sent = neg[i]
				sent = sent[0:len(sent)-1]
				if(sent[0]=='\t'):
					sent = sent[1:len(sent)-1]
			new_str+=sent

	
	train = []

	for i in range(800):
		train.append(corpus_pos[i])

	for i in range(800):
		train.append(corpus_neg[i])


	test = []

	for i in range(800,1000):
		test.append(corpus_pos[i])

	for i in range(800,1000):
		test.append(corpus_neg[i])

	
	dir_to_write =["Books","Dvd","Electronics","Kitchen"]
	original_dir = "../Dataset/Actualdata"+"/"+dir_to_write[num]+"/"

	train_file = original_dir+dir_to_write[num]+"train.txt"
	test_file = original_dir+dir_to_write[num]+"test.txt"


	
	with open(train_file,'w',encoding='utf-8') as f:
		for i in range(len(train)):
			f.write(train[i])
			f.write("\n")


	with open(test_file,'w',encoding='utf-8') as f:
		for i in range(len(test)):
			f.write(test[i])
			f.write("\n")
