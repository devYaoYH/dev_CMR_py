import sys
import math
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from CMR.params import Params
from CMR.learning import *
from semantic_model.DSM import Word2Vec
from visualization.Visualizer import Visualizer

VISUALIZE = False
NUM_SEQUENCES = 1
NUM_TRAILS_PER_SEQUENCE = 50

RANDOM_SEQUENCE = False

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
# print(wordlist)

# Load confound list
# confoundlist = ['anger', 'carefree', 'cheerful', 'tank', 'toothbrush', 'penguin']
confoundlist = ['fruit']
# confoundlist = []

# Recallable Vocabulary list
vocab = wordlist + confoundlist

# Load up pre-existing semantic knowledge base
semantic_kb = Word2Vec()

# RANDOM SAMPLE vector space
if (RANDOM_SEQUENCE):
	semantic_vocab = semantic_kb.get_vocab()
	wordlist = list(set(random.sample(semantic_vocab, 20)))
	print(wordlist)
	vocab = wordlist + confoundlist

# Initialize and seed parameters with wordlist
params = Params(wordlist, confound=confoundlist)			# Load Model parameters
visualizer = Visualizer(width=900, height=300)				# Init visualizer lib

# Initialize our Model
CMR = CMR_Model(params, semantic_kb, visualizer=visualizer if VISUALIZE else None)

serial_recall_counter = [0 for i in range(len(wordlist))]
confound_recall_counter = [0 for i in range(len(confoundlist))]
total_trials = 0

def run_trial(recall_samples=10):
	global total_trials
	# Learning Phase
	CMR.reset_network()
	CMR.begin_learning()
	li = np.random.permutation(len(wordlist))
	for i in li:
		rand_f = np.zeros((params.vocab_size, 1))
		rand_f[i] = 1
		CMR.present_feature(rand_f)
		# visualizer.update()
	CMR.end_learning()
		
	visualizer.update()
	print(li)
	print([vocab[i] for i in li])
	print("Finished Learning Phase")
	# visualizer.pause()

	# Recall Trails
	serial_position_to_idx = {li[i]: i for i in range(len(wordlist))}
	confound_position_to_idx = {i+len(wordlist): i for i in range(len(confoundlist))}
	progress_update = 0.1*recall_samples
	for t in range(recall_samples):
		if (t%progress_update == 0):
			print(f"Trail: {t+1}/{recall_samples}")
		recall_t = time.time()
		recall = CMR.recall(10000)

		# Update counter bins
		for r_id in recall:
			if (r_id in serial_position_to_idx):
				serial_recall_counter[serial_position_to_idx[r_id]] += 1
			elif (r_id in confound_position_to_idx):
				confound_recall_counter[confound_position_to_idx[r_id]] += 1

		visualizer.update()
		# print(recall)
		# print([vocab[r] for r in recall])
		# print("Finished Recall Phase")
	total_trials += recall_samples

# Test multiple learning trails
for t in range(NUM_SEQUENCES):
	run_trial(recall_samples=NUM_TRAILS_PER_SEQUENCE)

print("Finished Experiment")
visualizer.pause()
visualizer.exit()

# Give report
serial_graph = [s_c/total_trials for s_c in serial_recall_counter]
confound_graph = [c_c/total_trials for c_c in confound_recall_counter]

ax = plt.figure(1).gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

# plt.subplot(211)
plt.plot(serial_graph, linestyle="-", marker="2", color="g")
# plt.subplot(212)
plt.plot(confound_graph, linestyle="-", marker="2", color="r")

plt.show()
