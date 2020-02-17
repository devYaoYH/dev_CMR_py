class Params(object):
	def __init__(self, wordlist):
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
		self.vocab_size = len(self.vocab)