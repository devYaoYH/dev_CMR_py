import math
import time
import numpy as np
from .networks import CMR_Network

class CMR_Model(object):
	def __init__(self, params, visualizer=None):
		self.params = params
		self.network = CMR_Network(params, visualizer=visualizer)

	##################
	# LEARNING PHASE #
	##################
	# Presented with a feature vector (stimulus), train model network
	# M_fc (m x n) matrix		| Present within model network
	# f_i (n x 1) column vector	| Introduced as stimulus
	# recall (0, 1)	boolean		| Whether this is during recall
	def present_feature(self, f_i, recall=False):
		M_fc = self.network.M_fc
		c_in = np.matmul(M_fc, f_i)
		self.network.update_context(c_in, recall=recall)

	##################
	# ACCUM RECALLER #
	##################
	# Given a context vector, output a feature vector via accumulation process
	# M_cf (n x m) matrix		| Present within model network
	# c_i (m x 1) column vector	| Current context available to model
	# online (0, 1) boolean		| Is this a online model? (Recalled feature used
	#							  to modify context
	def recall_feature(self, c_i, online=False):
		M_cf = self.network.M_cf
		f_in = np.matmul(M_cf, c_i)
		f_out = self.network.recall(f_in)
		if (online):
			self.present_feature(f_out, recall=True)

	#############
	# MODEL I/O #
	#############
	# Dump model into binary data file to train/use later
	def dump(self):
		return True

	# Load model from binary data file
	def load(self, b_file):
		return True