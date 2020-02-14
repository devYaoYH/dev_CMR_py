class Params(object):
	def __init__(self):
		# Context Integration learning parameter
		self.c_beta = 0.1
		# Matrix_{FC} forward transition learning parameter
		self.M_fc_gamma = 0.6