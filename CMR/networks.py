import sys
import math
import numpy as np

class Network(object):

	# Layer class (Vector layer in network)
	class Layer(object):
		# Vector input -> Matrix of weights -> this layer
		def __init__(self, v):
			self.v = v
			self.has_weights = False

		def attach_weights(v_in, w_in, update_fn=None):
			# Sanity check for matrix size match
			if (type(v_in) is not np.ndarray or type(w_in) is not np.ndarray):
				raise ValueError("Not an numpy ndarray")
			v_in_shape = v_in.shape
			w_in_shape = w_in.shape
			if (w_in_shape[1] != v_in_shape[0]):
				raise ValueError(f"Incorrectly shaped Matrix: {w_in_shape} x {v_in_shape}")
			self.v_in = v_in
			self.w_in = w_in
			self.update_fn = update_fn
			self.has_weights = True

		def update(self):
			if (not self.has_weights):
				return True
			if (self.update_fn is not None):
				try:
					self.v = self.update_fn(self.v, self.v_in, self.w_in)
					return True
				except Exception as e:
					print(f"Layer update failed with Exception: {e}")
					return False
			else:
				self.v = np.matmul(self.w_in, self.v_in)
				return True

	def __init__(self, params, visualizer=None):
		self.params = params
		self.visualizer = visualizer

class CMR_Network(Network):
	def __init__(self, params, visualizer=None):
		super().__init__(params)
		# M_fc is a square matrix that encodes each feature
		# to context elements 1-to-1
		M_fc_size = (params.vocab_size, params.vocab_size)
		self.M_fc = self.init_weights(M_fc_size)
		
		# M_cf is a square matrix that encodes each context
		# to features
		M_cf_size = (params.vocab_size, params.vocab_size)
		self.M_cf = self.init_weights(M_cf_size)

		# Current context layer
		v_c_size = (params.vocab_size, 1)
		self.v_context = self.init_layer(v_c_size)

		# Feature layer
		v_f_size = (params.vocab_size, 1)
		self.v_feature = self.init_layer(v_f_size)

		# Attach feature to context
		self.v_feature.attach_weights(self.v_context, self.M_fc)

		# Attach context to feature
		self.v_context.attach_weights(self.v_feature, self.M_cf)

	def init_matrix(self, size):
		return np.zeros(size)

	def init_layer(self, size):
		return np.zeros(size)