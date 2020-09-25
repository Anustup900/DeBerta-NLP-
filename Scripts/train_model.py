import os
import re
import sklearn
import pandas as pd 
import numpy as np 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt 
from deBERTa import deberta
import copy
import torch
import os

import json
from .ops import *
from .bert import *
from .config import ModelConfig
from .cache_utils import load_model_state

__all__ = ['DeBERTa']
class DeBERTa(torch.nn.Module):
 def __init__(self, config=None, pre_trained=None):
    def classifier_train(chi_train_corpus_tf_idf,label_train,chi_test_corpus_tf_idf,label_test,train_choice,test_choice):

    rbf_parameters = [[0.9],[0.9],[0.9],[0.9],[0.9],[0.8],[0.9],[0.9],[0.8],[0.9],[0.9],[0.9]]

    val = (train_choice)*3 + test_choice

    Gamma = rbf_parameters[val][0]

    classifiers = [LogisticRegression(random_state=0),SVC(gamma=Gamma),DecisionTreeClassifier(random_state=0),
                   SVC(kernel='linear',gamma=Gamma),MultinomialNB(),KNeighborsClassifier(n_neighbors=7)]

    
    accu = []
    
    classify = ["LR","SVM-RBF","DT","SVM-L","MNB","KNN"]

    for i in range(len(classifiers)):
        acc,f1,cm = classify_sentiment(classifiers[i],chi_train_corpus_tf_idf,chi_test_corpus_tf_idf,label_train,label_test)

        accu.append(acc)

        print(classify[i]+" "+"F1 score is :",f1)
        print(classify[i]+" "+"confusion matrix is:")
        print(cm)
        print("\n")
        
       super().__init__()
    if config:
      self.z_steps = getattr(config, 'z_steps', 0)
    else:
      self.z_steps = 0

    state = None
    if pre_trained is not None:
      state, model_config = load_model_state(pre_trained)
      if config is not None and model_config is not None:
        for k in config.__dict__:
          if k not in ['hidden_size',
            'intermediate_size',
            'num_attention_heads',
            'num_hidden_layers',
            'vocab_size',
            'max_position_embeddings']:
            model_config.__dict__[k] = config.__dict__[k]
      config = copy.copy(model_config)
    self.embeddings = BertEmbeddings(config)
    self.encoder = BertEncoder(config)
    self.config = config
    self.pre_trained = pre_trained
    self.apply_state(state)
def classify_sentiment(classifier,chi_train_corpus_tf_idf,chi_test_corpus_tf_idf,label_train,label_test):
    clf   = classifier
    clf.fit(chi_train_corpus_tf_idf,label_train)
    pred = clf.predict(chi_test_corpus_tf_idf)
    accuracy = clf.score(chi_test_corpus_tf_idf,label_test)
    cm = confusion_matrix(pred,label_test)
    f1 = f1_score(pred,label_test)
    return accuracy,f1,cm 
def forward(self, input_ids, attention_mask=None, token_type_ids=None, output_all_encoded_layers=True, position_ids = None, return_att = False):
if attention_mask is None:
      attention_mask = torch.ones_like(input_ids)
    if token_type_ids is None:
      token_type_ids = torch.zeros_like(input_ids)

    embedding_output = self.embeddings(input_ids.to(torch.long), token_type_ids.to(torch.long), position_ids, attention_mask)
    encoded_layers = self.encoder(embedding_output,
                   attention_mask,
                   output_all_encoded_layers=output_all_encoded_layers, return_att = return_att)
    if return_att:
      encoded_layers, att_matrixs = encoded_layers

    if self.z_steps>1:
      hidden_states = encoded_layers[-2]
      layers = [self.encoder.layer[-1] for _ in range(z_steps)]
      query_states = encoded_layers[-1]
      rel_embeddings = self.encoder.get_rel_embedding()
      attention_mask = self.encoder.get_attention_mask(attention_mask)
      rel_pos = self.encoder.get_rel_pos(embedding_output)
      for layer in layers[1:]:
        query_states = layer(hidden_states, attention_mask, return_att=False, query_states = query_states, relative_pos=rel_pos, rel_embeddings=rel_embeddings)
        encoded_layers.append(query_states)

    if not output_all_encoded_layers:
      encoded_layers = encoded_layers[-1:]

    if return_att:
      return encoded_layers, att_matrixs
    return encoded_layers

 def apply_state(self, state = None):
  
    if self.pre_trained is None and state is None:
      return
    if state is None:
      state, config = load_model_state(self.pre_trained)
      self.config = config

    def key_match(key, s):
      c = [k for k in s if key in k]
      assert len(c)==1, c
      return c[0]
    current = self.state_dict()
    for c in current.keys():
      current[c] = state[key_match(c, state.keys())]
    self.load_state_dict(current)

    
