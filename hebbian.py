import sys
import math
import time
import random
import numpy as np
from CMR.params import Params
from CMR.learning import *
from semantic_model.DSM import Word2Vec
from visualization.Visualizer import Visualizer

##################
# TIMER FUNCTION #
##################
def debug_time(msg, init, now):
    print("{} {}ms".format(msg, int(round((now-init)*1000*1000))/1000.0), file=sys.stderr)

# Test hebbian learning + Visualize

# Load wordlist
fname = input("Wordlist: ")
wordlist = []
with open(f'data/{fname}.list') as fin:
	for line in fin:
		wordlist.append(line.strip())
print(wordlist)

# Load confound list
confoundlist = ['anger']

# Recallable Vocabulary list
vocab = wordlist + confoundlist

# Load up pre-existing semantic knowledge base
semantic_kb = Word2Vec()

# Initialize and seed parameters with wordlist
params = Params(wordlist, confound=confoundlist)			# Load Model parameters
visualizer = Visualizer(width=900, height=300)				# Init visualizer lib
CMR = CMR_Model(params, semantic_kb, visualizer=visualizer)	# Our CMR Model

# Learning Phase
li = np.random.permutation(len(wordlist))
for i in li:
	rand_f = np.zeros((params.vocab_size, 1))
	rand_f[i] = 1
	CMR.present_feature(rand_f)
	#visualizer.update()
	
visualizer.update()
print(li)
print([vocab[i] for i in li])
print("Finished Learning Phase")
visualizer.pause()

# Recall Phase
recall_t = time.time()
recall = CMR.recall()
debug_time("Recall Finished in:", recall_t, time.time())

visualizer.update()
print(recall)
print([vocab[r] for r in recall])
print("Finished Recall Phase")
visualizer.pause()