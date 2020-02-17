import sys
import math
import random
import numpy as np
from CMR.params import Params
from CMR.learning import *
from semantic_model.DSM import Word2Vec
from visualization.Visualizer import Visualizer

fname = input("Wordlist: ")

# Test hebbian learning + Visualize
wordlist = []
with open(f'data/{fname}.list') as fin:
	for line in fin:
		wordlist.append(line.strip())
print(wordlist)

# Load up pre-existing semantic knowledge base
semantic_kb = Word2Vec()

# Initialize and seed parameters with wordlist
params = Params(wordlist)									# Load Model parameters
visualizer = Visualizer(width=900, height=300)				# Init visualizer lib
CMR = CMR_Model(params, semantic_kb, visualizer=visualizer)	# Our CMR Model

# Learning Phase
li = np.random.permutation(params.vocab_size)
for i in li:
	rand_f = np.zeros((params.vocab_size, 1))
	rand_f[i] = 1
	CMR.present_feature(rand_f)
	visualizer.update()

print(li)
print("Finished Learning Phase")
visualizer.pause()

# Recall Phase
recall = CMR.recall()

visualizer.update()
print(recall)
print("Finished Recall Phase")
visualizer.pause()