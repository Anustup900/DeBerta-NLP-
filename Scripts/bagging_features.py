import os
import re
import sklearn
import pandas as pd 
import numpy as np 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score,confusion_matrix
from featureextraction import featureextraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest,chi2
from deBERTa import deberta

def MaskedLayerNorm(layerNorm, input, mask = None):
  """ Masked LayerNorm which will apply mask over the output of LayerNorm to avoid inaccurate updatings to the LayerNorm module.
  
  Args:
    layernorm (:obj:`~DeBERTa.deberta.BertLayerNorm`): LayerNorm module or function
    input (:obj:`torch.tensor`): The input tensor
    mask (:obj:`torch.IntTensor`): The mask to applied on the output of LayerNorm where `0` indicate the output of that element will be ignored, i.e. set to `0`

  Example::

    # Create a tensor b x n x d
    x = torch.randn([1,10,100])
    m = torch.tensor([[1,1,1,0,0,0,0,0,0,0]], dtype=torch.int)
    LayerNorm = DeBERTa.deberta.BertLayerNorm(100)
    y = MaskedLayerNorm(LayerNorm, x, m)

  """
 from sklearn.ensemble import BaggingClassifier

lr_parameters = [[7,0.5],[8,0.5],[6,0.5],[10,0.8],[8,0.6],[14,0.9],[7,0.8],[8,0.8],[3,0.8],[5,0.7],[2,0.5],[5,0.9]]
rbf_parameters = [[8,0.8],[9,0.6],[13,0.7],[5,0.6],[7,0.5],[5,0.6],[5,0.6],[12,0.8],[2,0.6],[5,0.8],[8,0.8],[5,0.8]]
dt_parameters = [[8,0.8],[9,0.5],[15,0.5],[15,0.9],[8,0.9],[10,0.7],[15,0.9],[25,0.8],[15,0.9],[5,0.9],[15,0.9],[15,0.9]]
linear_parameters = [[8,0.5],[10,0.5],[7,0.9],[10,0.5],[13,0.6],[5,0.5],[7,0.8],[9,0.9],[3,0.4],[5,0.8],[5,0.9],[5,0.8]]
nb_parameters = [[8,0.5],[3,0.6],[5,0.7],[2,0.5],[5,0.9],[5,0.9],[3,0.9],[8,0.8],[5,0.9],[11,0.6],[3,0.9],[3,0.9]]
knn_parameters = [[7,0.5],[7,0.6],[7,0.5],[9,0.6],[15,0.8],[15,0.9],[10,0.8],[10,0.8],[15,0.9],[5,0.8],[10,0.9],[10,0.8]]



def bagging_classify_sentiment(classifier,chi_train_corpus_tf_idf,chi_test_corpus_tf_idf,label_train,label_test,num,samples):
    
    clf   = BaggingClassifier(base_estimator=classifier,random_state=0,max_samples=samples,n_estimators=num)
    clf.fit(chi_train_corpus_tf_idf,label_train)
    pred = clf.predict(chi_test_corpus_tf_idf)
    accuracy = clf.score(chi_test_corpus_tf_idf,label_test)
    cm = confusion_matrix(pred,label_test)
    f1 = f1_score(pred,label_test)
    return accuracy,f1,cm 


def bagging_train(chi_train_corpus_tf_idf,label_train,chi_test_corpus_tf_idf,label_test,train_choice,test_choice):

	x = (train_choice)*3 + test_choice

	rbf_function = [[0.9],[0.9],[0.9],[0.9],[0.9],[0.8],[0.9],[0.9],[0.8],[0.9],[0.9],[0.9]]

	Gamma = rbf_function[x][0]

	classify = ["LR","SVM-RBF","DT","SVM-L","MNB","KNN"]



	classifiers = [LogisticRegression(random_state=0),SVC(gamma=Gamma),DecisionTreeClassifier(random_state=0),SVC(kernel='linear',gamma=Gamma),MultinomialNB(),KNeighborsClassifier(n_neighbors=7)]

	param = [lr_parameters[x],rbf_parameters[x],dt_parameters[x],linear_parameters[x],nb_parameters[x],knn_parameters[x]]

	for i in range(len(classifiers)):
		p = param[i]
		print(p)
		num = p[0]
		samples = p[1]
		acc,f1,cm = bagging_classify_sentiment(classifiers[i],chi_train_corpus_tf_idf,chi_test_corpus_tf_idf,label_train,label_test,num,samples)


		
		print("Bagging "+classify[i]+" "+"F1 score is :",f1)
		print("Bagging "+classify[i]+" "+"confusion matrix is:")
		print(cm)
		print("\n")
    output = layerNorm(input).to(input)
  if mask is None:
    return output
  if mask.dim()!=input.dim():
    if mask.dim()==4:
      mask=mask.squeeze(1).squeeze(1)
    mask = mask.unsqueeze(2)
  mask = mask.to(output.dtype)
  return output*mask
