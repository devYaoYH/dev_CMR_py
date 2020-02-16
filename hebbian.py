import sys
import math
import numpy as np
from CMR.params import Params
from CMR.learning import *
from visualization.Visualizer import Visualizer

# Test hebbian learning + Visualize
params = Params()			# Load Model parameters
visualizer = Visualizer()	# Init visualizer lib
CMR = CMR_Model(params)		# Our CMR Model

l1_size = (30, 30)
l1 = visualizer.add_layer(l1_size)

l2_size = (30, 8)
l2 = visualizer.add_layer(l2_size)

for x in range(4):
	l1.update_data(np.random.uniform(-1, 1, l1_size))
	l2.update_data(np.random.uniform(-1, 1, l2_size))
	visualizer.update()
	visualizer.pause()