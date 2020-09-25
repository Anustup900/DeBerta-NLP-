import copy
import torch
from torch import nn
from collections import Sequence
from packaging import version
import numpy as np
import math
import os
import pdb

import json
from .ops import *
from .disentangled_attention import *

import os
import re
import sklearn
import pandas as pd 
import numpy as np 

from DeBERTa import deberta
class BertEmbeddings(nn.Module):
  """Construct the embeddings from word, position and token_type embeddings.
  """
  def __init__(self, config):
    super(BertEmbeddings, self).__init__()
    padding_idx = getattr(config, 'padding_idx', 0)
    self.embedding_size = getattr(config, 'embedding_size', config.hidden_size)
    self.word_embeddings = nn.Embedding(config.vocab_size, self.embedding_size, padding_idx = padding_idx)

    self.position_biased_input = getattr(config, 'position_biased_input', True)
    if not self.position_biased_input:
      self.position_embeddings = None
    else:
      self.position_embeddings = nn.Embedding(config.max_position_embeddings, self.embedding_size)

    if config.type_vocab_size>0:
      self.token_type_embeddings = nn.Embedding(config.type_vocab_size, self.embedding_size)
    
    if self.embedding_size != config.hidden_size:
      self.embed_proj = nn.Linear(self.embedding_size, config.hidden_size, bias=False)
    self.LayerNorm = BertLayerNorm(config.hidden_size, config.layer_norm_eps)
    self.dropout = StableDropout(config.hidden_dropout_prob)
    self.output_to_half = False
    self.config = config

  def forward(self, input_ids, token_type_ids=None, position_ids=None, mask = None):
    seq_length = input_ids.size(1)
    if position_ids is None:
      position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
      position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    if token_type_ids is None:
      token_type_ids = torch.zeros_like(input_ids)

    words_embeddings = self.word_embeddings(input_ids)
    if self.position_embeddings is not None:
      position_embeddings = self.position_embeddings(position_ids.long())
    else:
      position_embeddings = torch.zeros_like(words_embeddings)

    embeddings = words_embeddings
    if self.position_biased_input:
      embeddings += position_embeddings
    if self.config.type_vocab_size>0:
      token_type_embeddings = self.token_type_embeddings(token_type_ids)
      embeddings += token_type_embeddings

    if self.embedding_size != self.config.hidden_size:
      embeddings = self.embed_proj(embeddings)

    embeddings = MaskedLayerNorm(self.LayerNorm, embeddings, mask)
    embeddings = self.dropout(embeddings)
    return embeddings
    
  val = (train_choice)*3 + test_choice

	k = chi_square_parameters[val][0]

	if(k=='all'):
		K = train_corpus_tf_idf.shape[1]
	else:
		K = k 

	vectorizer_chi2 = SelectKBest(chi2,k=K)

	chi_train_corpus_tf_idf = vectorizer_chi2.fit_transform(train_corpus_tf_idf,label_train)

	chi_test_corpus_tf_idf = vectorizer_chi2.transform(test_corpus_tf_idf)

	return [chi_train_corpus_tf_idf,chi_test_corpus_tf_idf]
