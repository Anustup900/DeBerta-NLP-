import os
import re
import sklearn
import pandas as pd 
import numpy as np 
from deBERTa import deberta

class BertEncoder(nn.Module):

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

  def __init__(self, config):
    super().__init__()
    layer = BertLayer(config)
    self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])
    self.relative_attention = getattr(config, 'relative_attention', False)
    if self.relative_attention:
      self.max_relative_positions = getattr(config, 'max_relative_positions', -1)
      if self.max_relative_positions <1:
        self.max_relative_positions = config.max_position_embeddings
      self.rel_embeddings = nn.Embedding(self.max_relative_positions*2, config.hidden_size)
  def get_rel_embedding(self):
    rel_embeddings = self.rel_embeddings.weight if self.relative_attention else None
    return rel_embeddings

  def get_attention_mask(self, attention_mask):
    if attention_mask.dim()<=2:
      extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
      attention_mask = extended_attention_mask*extended_attention_mask.squeeze(-2).unsqueeze(-1)
      attention_mask = attention_mask.byte()
    elif attention_mask.dim()==3:
      attention_mask = attention_mask.unsqueeze(1)

    return attention_mask

  def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
    if self.relative_attention and relative_pos is None:
      q = query_states.size(-2) if query_states is not None else hidden_states.size(-2)
      relative_pos = build_relative_position(q, hidden_states.size(-2), hidden_states.device)
    return relative_pos

  def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, return_att=False, query_states = None, relative_pos=None):
    attention_mask = self.get_attention_mask(attention_mask)
    relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)

    all_encoder_layers = []
    att_matrixs = []
    if isinstance(hidden_states, Sequence):
      next_kv = hidden_states[0]
    else:
      next_kv = hidden_states
    rel_embeddings = self.get_rel_embedding()
    for i, layer_module in enumerate(self.layer):
      output_states = layer_module(next_kv, attention_mask, return_att, query_states = query_states, relative_pos=relative_pos, rel_embeddings=rel_embeddings)
      if return_att:
        output_states, att_m = output_states

      if query_states is not None:
        query_states = output_states
        if isinstance(hidden_states, Sequence):
          next_kv = hidden_states[i+1] if i+1 < len(self.layer) else None
      else:
        next_kv = output_states

      if output_all_encoded_layers:
        all_encoder_layers.append(output_states)
        if return_att:
          att_matrixs.append(att_m)
    if not output_all_encoded_layers:
      all_encoder_layers.append(output_states)
      if return_att:
        att_matrixs.append(att_m)
    if return_att:
      return (all_encoder_layers, att_matrixs)
    else:
      return all_encoder_layers
   
  query_states =  layer_module.fit_transform(query_states,label_train)

	 relative_pos=  layer_module.transform( relative_pos)

	return [relative_pos,query_states]
