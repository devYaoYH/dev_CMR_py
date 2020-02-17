import sys
import math
import numpy as np

class Network(object):

	# Layer class (Vector layer in network) | SAME AS WEIGHT?
	class Layer(object):
		# Vector input -> Matrix of weights -> this layer
		def __init__(self, v, vis=None):
			self.v = v
			self.shape = v.shape
			self.vis = vis
			# self.has_weights = False

		def add_visualizer(self, vis_obj):
			self.vis = vis_obj

		def update(self, v):
			if (v.shape != self.shape):
				raise ValueError(f"Size mismatch new_v:{v.shape} | {self.shape}")
			self.v = v
			if (self.vis is not None):
				self.vis.update(self.v)

		# def attach_weights(v_in, w_in, update_fn=None):
		# 	# Sanity check for matrix size match
		# 	if (not isinstance(v_in, self.Layer) or not isinstance(w_in, self.Weight)):
		# 		raise ValueError("Not an Network Objects")
		# 	v_in_shape = v_in.shape
		# 	w_in_shape = w_in.shape
		# 	if (w_in_shape[1] != v_in_shape[0]):
		# 		raise ValueError(f"Incorrectly shaped Matrix: {w_in_shape} x {v_in_shape}")
		# 	self.v_in = v_in
		# 	self.w_in = w_in
		# 	self.update_fn = update_fn
		# 	self.has_weights = True

		# def detach_weights(self):
		# 	self.has_weights = False

		# def update(self):
		# 	if (not self.has_weights):
		# 		return True
		# 	if (self.update_fn is not None):
		# 		try:
		# 			self.v = self.update_fn(self.v, self.v_in, self.w_in)
		# 			if (self.vis is not None):
		# 				self.vis.update(self.v)
		# 			return True
		# 		except Exception as e:
		# 			print(f"Layer update failed with Exception: {e}")
		# 			return False
		# 	else:
		# 		self.v = np.matmul(self.w_in, self.v_in)
		# 		if (self.vis is not None):
		# 			self.vis.update(self.v)
		# 		return True

	# Weight class (Layer - Layer connection weights)
	class Weight(object):
		def __init__(self, w, vis=None):
			self.w = w
			self.shape = w.shape
			self.vis = vis

		def add_visualizer(self, vis_obj):
			self.vis = vis_obj

		def update(self, w):
			if (w.shape != self.shape):
				raise ValueError(f"Size mismatch new_w:{w.shape} | {self.shape}")
			self.w = w
			if (self.vis is not None):
				self.vis.update(self.w)

	def __init__(self, params, visualizer=None):
		self.params = params
		self.visualizer = visualizer

	def update_context(self, stimulus, recall=False):
		return stimulus

class CMR_Network(Network):
	def __init__(self, params, semantic_kb, visualizer=None):
		super().__init__(params, visualizer=visualizer)

		# Semantic knowledge base (loaded model of word2vec | LSA etc...)
		self.semantic_kb = semantic_kb

		# M_fc_pre is a square identity matrix encoding pre-existing
		# associations from features to context
		self.M_fc_pre = np.identity(params.vocab_size)

		# M_cf_pre is a square matrix with semantic encodings
		self.M_cf_pre = self.generate_semantics(params.vocab)
		
		# M_fc is a square matrix that encodes each feature
		# to context elements 1-to-1
		M_fc_size = (params.vocab_size, params.vocab_size)
		self.M_fc = self.init_weights(M_fc_size)
		self.M_fc.update((1-params.M_fc_gamma)*self.M_fc_pre)

		# Current context layer
		v_c_size = (params.vocab_size, 1)
		self.v_context = self.init_layer(v_c_size)
		
		# M_cf is a square matrix that encodes each context
		# to features
		M_cf_size = (params.vocab_size, params.vocab_size)
		self.M_cf = self.init_weights(M_cf_size)
		self.M_cf.update(params.M_cf_pre_scaling*self.M_cf_pre)

		# Feature recall from context
		v_recall_size = (params.vocab_size, 1)
		self.v_recall = self.init_layer(v_recall_size)

		# Accumulator layer
		v_accum_size = (params.vocab_size, 1)
		self.v_accumulator = self.init_layer(v_accum_size)

		# Feature output layer
		v_f_size = (params.vocab_size, 1)
		self.v_feature = self.init_layer(v_f_size)

		#######################
		# Perhaps not needed? #
		#######################
		# Attach feature to context
		# self.v_feature.attach_weights(self.v_context, self.M_fc)

		# Attach context to feature
		# self.v_context.attach_weights(self.v_feature, self.M_cf)

	def init_weights(self, size):
		#TODO Anymore inits necessary for creation of weight matrix?
		vis_obj = None
		if (self.visualizer is not None):
			vis_obj = self.visualizer.add_layer(size)
			print("adding layer to visualizer")
		return self.Weight(np.zeros(size), vis=vis_obj)

	def init_layer(self, size):
		#TODO Anymore inits necessary for creation of network layer?
		vis_obj = None
		if (self.visualizer is not None):
			vis_obj = self.visualizer.add_layer(size)
			print("adding layer to visualizer")
		return self.Layer(np.zeros(size), vis=vis_obj)

	def generate_semantics(self, wordlist):
		sem_mat = np.zeros((len(wordlist), len(wordlist)))
		for i, w1 in enumerate(wordlist):
			for j, w2 in enumerate(wordlist):
				sem_mat[i,j] = self.semantic_kb.get_distance(w1, w2)
		return sem_mat

	def update_context(self, stimulus, recall=False):
		#TODO Pad stimulus if incorrectly shaped?
		c_in = np.matmul(self.M_fc.w, stimulus)
		c_cur = self.v_context.v
		beta = self.params.c_beta_rec if recall else self.params.c_beta
		rho = math.sqrt(1 + (beta**2)*(np.dot(c_cur.transpose(), c_in)**2 - 1))
		rho -= beta * np.dot(c_cur.transpose(), c_in)
		c_i = rho*c_cur + beta*c_in
		self.v_context.update(c_i)
		if (not recall):
			# Learn and update M_fc
			delta_M_fc = np.matmul(c_i, stimulus.transpose())
			# new_M_fc = (1-self.params.M_fc_gamma)*self.M_fc.w + self.params.M_fc_gamma*delta_M_fc
			new_M_fc = self.M_fc.w + self.params.M_fc_gamma*delta_M_fc
			self.M_fc.update(new_M_fc)
			# Learn and update M_cf
			delta_M_cf = np.matmul(stimulus, c_i.transpose())
			# new_M_cf = (1-self.params.M_cf_gamma)*self.M_cf.w + self.params.M_cf_gamma*delta_M_cf
			new_M_cf = self.M_cf.w + self.params.M_cf_gamma*delta_M_cf
			self.M_cf.update(new_M_cf)

	def get_current_context(self):
		return self.v_context.v

	def recall(self, context):
		#TODO Pad features if more required?
		f_in = np.matmul(self.M_cf.w, context)
		self.v_recall.update(f_in)
		#TODO IMPT Implement Accumulator design
		f_out = np.zeros(f_in.shape)
		for i in range(f_in.shape[0]):
			f_in[i] = np.random.uniform(0, f_in[i])
		max_i = 0
		max_v = 0
		for i in range(f_in.shape[0]):
			if (f_in[i] > max_v):
				max_v = f_in[i]
				max_i = i
		f_out[max_i] = 1
		return f_out
