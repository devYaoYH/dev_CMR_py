import sys
import math
import numpy as np

class Network(object):
	# Layer class (Vector layer in network)
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

	def __init__(self, params, visualizer=None):
		self.params = params
		self.visualizer = visualizer

	def update_context(self, stimulus, recall=False):
		return stimulus

	def recall(self, context):
		return context

class CMR_Network(Network):
	def __init__(self, params, semantic_kb, visualizer=None):
		super().__init__(params, visualizer=visualizer)

		# Accumulator Parameters
		self.accum_noise_sd = params.subject_noise_sd
		self.accum_thresh = params.accum_thresh
		self.accum_tau = params.accum_tau
		self.accum_ki = params.accum_ki
		self.accum_lambda = params.accum_lambda
		self.accum_inhib = np.add(-1*np.identity(params.vocab_size), 1)

		# Accumulator Time Keeping
		self.accum_timer = 0
		self.accum_timeout = 0

		# Semantic knowledge base (loaded model of word2vec | LSA etc...)
		self.semantic_kb = semantic_kb

		# M_fc_pre is a square identity matrix encoding pre-existing
		# associations from features to context
		self.M_fc_pre = np.identity(params.vocab_size)

		# M_cf_pre is a square matrix with semantic encodings
		self.M_cf_pre = self.generate_semantics(params.vocab + params.confound)
		
		# M_fc is a square matrix that encodes each feature
		# to context elements 1-to-1
		M_fc_size = (params.vocab_size, params.vocab_size)
		self.M_fc = self.init_layer(M_fc_size)
		self.M_fc.update((1-params.M_fc_gamma)*self.M_fc_pre)

		# Current context layer
		v_c_size = (params.vocab_size, 1)
		self.v_context = self.init_layer(v_c_size)
		
		# M_cf is a square matrix that encodes each context
		# to features
		M_cf_size = (params.vocab_size, params.vocab_size)
		self.M_cf = self.init_layer(M_cf_size)
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
		c_in = np.matmul(self.M_fc.v, stimulus)
		c_cur = self.v_context.v
		beta = self.params.c_beta_rec if recall else self.params.c_beta
		rho = math.sqrt(1 + (beta**2)*(np.dot(c_cur.transpose(), c_in)**2 - 1))
		rho -= beta * np.dot(c_cur.transpose(), c_in)
		c_i = rho*c_cur + beta*c_in
		self.v_context.update(c_i)
		if (not recall):
			# Learn and update M_fc
			delta_M_fc = np.matmul(c_i, stimulus.transpose())
			# new_M_fc = (1-self.params.M_fc_gamma)*self.M_fc.v + self.params.M_fc_gamma*delta_M_fc
			new_M_fc = self.M_fc.v + self.params.M_fc_gamma*delta_M_fc
			self.M_fc.update(new_M_fc)
			# Learn and update M_cf
			delta_M_cf = np.matmul(stimulus, c_i.transpose())
			# new_M_cf = (1-self.params.M_cf_gamma)*self.M_cf.v + self.params.M_cf_gamma*delta_M_cf
			new_M_cf = self.M_cf.v + self.params.M_cf_gamma*delta_M_cf
			self.M_cf.update(new_M_cf)

	def get_current_context(self):
		return self.v_context.v

	# Generates a uniform distribution of noise given a size
	# based on network noise parameter
	def random_noise(self, size):
		return np.random.normal(0, self.accum_noise_sd, size)

	# Scans through accumulators to find signal above threshold
	def select_winner(self, accumulator, thresh=None):
		# Default threshold is 1
		if (thresh is None):
			thresh = self.accum_thresh
		# Sums along row (each row activates a feature signal)
		#accum = accumulator.sum(axis=1)
		accum = accumulator

		# Looks for max activated signal
		max_i = 0
		max_v = 0
		for i in range(accum.shape[0]):
			if (accum[i] > max_v):
				max_v = accum[i]
				max_i = i
		if (max_v > thresh):
			return max_i
		else:
			return None

	# Returns the index of winner row in accumulator banks
	# MUTATES accumulator
	# Timer increments by tau each step (tau: step-size)
	# Timouts on timer exceeding timeout
	def random_accumulator(self, f_in, accumulator, tau, timer, timeout):
		for i in range(f_in.shape[0]):
			accumulator[i] = np.random.uniform(0, f_in[i])
		winner = self.select_winner(accumulator)
		while (timer < timeout and winner is None):
			timer += tau
			accumulator += tau*self.random_noise(accumulator.shape)
			winner = self.select_winner(accumulator)
			self.v_accumulator.update(accumulator)
		return winner, timer

	# Competitibe Inhibitory accumulator design
	def inhibitory_accumulator(self, f_in, accumulator, tau, timer, timeout):
		print(accumulator.shape)
		print(-tau*self.accum_lambda*self.accum_inhib)
		print(1 - tau*self.accum_ki)
		cur_inhib = 1 - tau*self.accum_ki - tau*self.accum_lambda*self.accum_inhib
		print(cur_inhib)
		winner = None
		while (timer < timeout and winner is None):
			timer += tau
			# Inhibition from other signals
			accumulator = np.matmul(cur_inhib,accumulator)
			# Accumulate f_in
			accumulator += tau*f_in
			# Add noise
			accumulator += self.random_noise(accumulator.shape)
			# Clamps val to non-negative
			for i in range(accumulator.shape[0]):
				accumulator[i] = max(0, accumulator[i])
			self.v_accumulator.update(accumulator)
		return winner, timer

	def start_accumulator_for_timeout(self, timeout):
		self.accum_timer = 0
		self.accum_timeout = timeout

	def recall(self, context):
		#TODO Pad features if more required?
		f_in = np.matmul(self.M_cf.v, context)
		self.v_recall.update(f_in)

		#TODO IMPT Implement Accumulator design
		f_out = np.zeros(f_in.shape)

		# RANDOM Accumulator based on f_in values
		# cur_recall, self.accum_timer = self.random_accumulator(f_out, f_in, self.accum_tau, self.accum_timer, self.accum_timeout)
		
		# INHIBITORY Accumulator
		cur_recall, self.accum_timer = self.inhibitory_accumulator(f_out, f_in, self.accum_tau, self.accum_timer, self.accum_timeout)

		# Observe recall result (whether is timeout)
		if (cur_recall is None):
			self.accum_timer = 999999
			return None, None
		
		f_recall = np.zeros(f_out.shape)
		f_recall[cur_recall] = 1

		#DISPLAY
		self.v_feature.update(f_recall)
		
		return cur_recall, f_recall