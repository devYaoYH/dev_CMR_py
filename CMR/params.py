class Params(object):
	def __init__(self, wordlist, confound=None):
		# Network Noise (Subject simulation with ~N(0, sd^2))
		self.subject_noise_sd = 0.15
		# Accumulator Threshold (Firing limit)
		self.accum_thresh = 1
		self.accum_tau = 0.24	# Accumulator time step-size
		self.accum_ki = 0.1		# Accumulator decay rate
		self.accum_lambda = 0.38# Accumulator lateral inhibition scaling
		# Context Integration learning parameter
		self.c_beta = 0.75
		self.c_beta_rec = 0.3
		# Matrix_{FC} forward transition learning parameter
		self.M_fc_gamma = 0.6
		# Matrix_{CF} learning parameter
		self.M_cf_pre_scaling = 1.8
		self.M_cf_gamma = 1
		# Word list to test for recall
		self.vocab = wordlist
		self.confound = confound
		self.vocab_size = len(self.vocab) + len(self.confound)