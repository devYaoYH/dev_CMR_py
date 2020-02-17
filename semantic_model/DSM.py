import numpy as np
from scipy.spatial.distance import cosine
from gensim.models.keyedvectors import KeyedVectors

class Semantic_KB(object):
	def __init__(self):
		self.model = None

	def get_distance(self, w1, w2):
		# Get word index from model
		return 1

class Word2Vec(Semantic_KB):
	def __init__(self):
		# Load up restricted wordset from Word2Vec
		self.model = KeyedVectors.load('C:/_YaoYiheng/Projects/dev_CMR_py/data/restricted_word2vec.bin')

	def get_distance(self, w1, w2):
		allowable = self.model.vocab.keys()
		if (w1 not in allowable):
			raise ValueError(f"Word: {w1} not in loaded vocab")
		if (w2 not in allowable):
			raise ValueError(f"Word: {w2} not in loaded vocab")
		return 1-cosine(self.model[w1], self.model[w2])