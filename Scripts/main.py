import os
import re
import sys
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
from train import *
from featureextraction import *
from featureselection import *
from preprocess import *
from bagging import *

class ModelConfig(AbsModelConfig):   //deberta model configuration
 """Configuration class to store the configuration of a :class:`~DeBERTa.deberta.DeBERTa` model.

        Attributes:
            hidden_size (int): Size of the encoder layers and the pooler layer, default: `768`.
            num_hidden_layers (int): Number of hidden layers in the Transformer encoder, default: `12`.
            num_attention_heads (int): Number of attention heads for each attention layer in
                the Transformer encoder, default: `12`.
            intermediate_size (int): The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder, default: `3072`.
            hidden_act (str): The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported, default: `gelu`.
            hidden_dropout_prob (float): The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler, default: `0.1`.
            attention_probs_dropout_prob (float): The dropout ratio for the attention
                probabilities, default: `0.1`.
            max_position_embeddings (int): The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048), default: `512`.
            type_vocab_size (int): The vocabulary size of the `token_type_ids` passed into
                `DeBERTa` model, default: `-1`.
            initializer_range (int): The sttdev of the _normal_initializer for
                initializing all weight matrices, default: `0.02`.
            relative_attention (:obj:`bool`): Whether use relative position encoding, default: `False`.
            max_relative_positions (int): The range of relative positions [`-max_position_embeddings`, `max_position_embeddings`], default: -1, use the same value as `max_position_embeddings`. 
            padding_idx (int): The value used to pad input_ids, default: `0`.
            position_biased_input (:obj:`bool`): Whether add absolute position embedding to content embedding, default: `True`.
            pos_att_type (:obj:`str`): The type of relative position attention, it can be a combination of [`p2c`, `c2p`, `p2p`], e.g. "p2c", "p2c|c2p", "p2c|c2p|p2p"., default: "None".


    """
    
     def __init__(self):
        """Constructs ModelConfig.

        """
        
        self.hidden_size = 768
        self.num_hidden_layers = 12
        self.num_attention_heads = 12
        self.hidden_act = "gelu"
        self.intermediate_size = 3072
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = 512
        self.type_vocab_size = 0
        self.initializer_range = 0.02
        self.layer_norm_eps = 1e-7
        self.padding_idx = 0
        self.vocab_size = -1



def getvalue(val):
	if(val=="books"):
		return 0
	elif(val=="dvd"):
		return 1
	elif(val=="electronics"):
		return 2
	else:
		return 3

def getchoices(train,test):

	if(train=="books"):
		train_choice = 0
		if(test=="dvd"):
			test_choice = 0
		elif(test=="electronics"):
			test_choice = 1
		else:
			test_choice = 2

	if(train=="dvd"):
		train_choice = 1
		if(test=="books"):
			test_choice = 0
		elif(test=="electronics"):
			test_choice = 1
		else:
			test_choice = 2

	if(train=="electronics"):
		train_choice = 2
		if(test=="books"):
			test_choice = 0
		elif(test=="dvd"):
			test_choice = 1
		else:
			test_choice = 2

	if(train=="kitchen"):
		train_choice = 3
		if(test=="books"):
			test_choice = 0
		elif(test=="dvd"):
			test_choice = 1
		else:
			test_choice = 2

	return train_choice,test_choice


if __name__ == "__main__":

	train = sys.argv[1]

	test = sys.argv[2]

	
	train_choice,test_choice = getchoices(train,test)

	pre_choice_train = getvalue(train)
	pre_choice_test = getvalue(test)

	corpus_train,label_train,corpus_test,label_test = preprocessing(pre_choice_train,pre_choice_test)

	
	train_corpus_tf_idf,test_corpus_tf_idf = featureextraction(corpus_train,corpus_test,label_train,train_choice,test_choice)

	chi_train_corpus_tf_idf,chi_test_corpus_tf_idf = featureselection(train_corpus_tf_idf,test_corpus_tf_idf,label_train,train_choice,test_choice)

	classifier_train(chi_train_corpus_tf_idf,label_train,chi_test_corpus_tf_idf,label_test,train_choice,test_choice)

	bagging_train(chi_train_corpus_tf_idf,label_train,chi_test_corpus_tf_idf,label_test,train_choice,test_choice)
