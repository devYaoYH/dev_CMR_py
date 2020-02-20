import math
import time
import numpy as np
from .networks import CMR_Network

class CMR_Model(object):
	def __init__(self, params, semantic_kb, visualizer=None):
		self.params = params
		self.semantic_kb = semantic_kb
		self.feature_in = None
		self.visualizer = visualizer
		if (self.visualizer is not None):
			self.feature_in = self.visualizer.add_layer((self.params.vocab_size, 1))
		self.network = CMR_Network(params, semantic_kb, visualizer=visualizer)

	##################
	# LEARNING PHASE #
	##################
	# Presented with a feature vector (stimulus), train model network
	# M_fc (m x n) matrix		| Present within model network (CMR)
	# f_i (n x 1) column vector	| Introduced as stimulus
	# recall (0, 1)	boolean		| Whether this is during recall
	def present_feature(self, f_i, recall=False):
		if (self.feature_in is not None):
			self.feature_in.update(f_i)
		return self.network.update_context(f_i, recall=recall)

	##################
	# ACCUM RECALLER #
	##################
	# Given a context vector, output a feature vector via accumulation process
	# M_cf (n x m) matrix		| Present within model network (CMR)
	# c_i (m x 1) column vector	| Current context available to model
	# online (0, 1) boolean		| Is this a online model? (Recalled feature used
	#							  to modify context
	def recall_feature(self, c_i, online=False):
		f_idx, f_out = self.network.recall(c_i)
		if (online and f_out is not None):
			self.present_feature(f_out, recall=True)
		return f_idx

	#TODO Needs to be implemented using time-limited/count limited accumulator design
	def recall(self):
		c_i = self.network.get_current_context()
		self.network.start_accumulator_for_timeout(90000)
		recall_idx = -1
		recall_series = []
		inputs_max_idx = len(self.params.vocab)
		inputs_left = inputs_max_idx
		while (recall_idx is not None and inputs_left > 0):
			print(inputs_left)
			recall_idx = self.recall_feature(c_i, online=True)
			if (recall_idx is None):
				return recall_series
			if (recall_idx < inputs_max_idx):
				inputs_left -= 1
			recall_series.append(recall_idx)
			self.visualizer.update()
		return recall_series

	#############
	# MODEL I/O #
	#############
	# Dump model into binary data file to train/use later
	def dump(self):
		return True

	# Load model from binary data file
	def load(self, b_file):
		return True