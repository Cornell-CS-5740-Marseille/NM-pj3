import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import csv
from pathlib import Path 
from model import model
import time
from tqdm import tqdm

def read_input(task):
	assert task in {"copy", "reverse", "sort"}
	# modify the path
	relativePath = "toy_data/" + "toy_" + task + "/"
	path = Path(relativePath)

	train_f = path / ("train/sources.txt")
	test_f = path / ("test/sources.txt")
	
	with open(train_f, encoding='utf-8', errors='ignore') as train_file:
		train_inputs = [line.split() for line in train_file]
	train_file.close()
	
	with open(test_f, encoding='utf-8', errors='ignore') as test_file:
		test_inputs = [line.split() for line in test_file]
	test_file.close()
	
	if task == "copy":
		train_outputs, test_outputs = train_inputs, test_inputs
	elif task == "reverse":
		train_outputs = [seq[::-1] for seq in train_inputs]
		test_outputs = [seq[::-1] for seq in test_inputs]
	else:
		train_outputs = [sorted(seq) for seq in train_inputs]
		test_outputs = [sorted(seq) for seq in test_inputs]
	return [train_inputs, train_outputs, test_inputs, test_outputs]


def build_indices(train_set):
	tokens = [token for line in train_set for token in line]

	# From token to its index
	forward_dict = {'UNK': 0}

	# From index to token
	backward_dict = {0: 'UNK'}
	i = 1
	for token in tokens:
		if token not in forward_dict:
			forward_dict[token] = i 
			backward_dict[i] = token 
			i += 1
	return forward_dict, backward_dict


def encode(data, forward_dict):
	return [list(map(lambda t: forward_dict.get(t,0), line)) for line in data]


def saveLoss(data_dict):
	root = "test/" + varyParameters + "/metrics/"
	fname = root + "exp_%s.csv" % (varyParameters)
	with open(fname, 'w', newline='') as csvfile:
		losswriter = csv.writer(csvfile, delimiter=',',
								quotechar=',', quoting=csv.QUOTE_MINIMAL)
		losswriter.writerow(['sequence', 'loss', 'time', 'dataSet', 'num_examples', 'vocab_size', 'max_len', 'special_num'])
		for dataset in data_dict:
			for value in data_dict[dataset]:
				loss_list = data_dict[dataset][value]["loss"]
				minibatch_size = data_dict[dataset][value]["batch"]
				timer_list = data_dict[dataset][value]["timer"]
				for x in range(len(loss_list)):
					sequence_num = (x + 1) * minibatch_size
					cost_per_sequence = loss_list[x]
					time = timer_list[x]
					losswriter.writerow([sequence_num, cost_per_sequence, time, dataset, int(value), vocab_size, max_len, 5])

def saveLog(reference, candidate, log):
	root = "test/" + varyParameters + "/logs/" + data_type + "/"
	if not os.path.exists(root):
		os.mkdir(root)
	rname = root + "reference_%s.csv" % (current_value)
	cname = root + "candidate_%s.csv" % (current_value)

	with open(rname, 'w') as f:
		for item in reference:
			item = map(lambda x: str(x), item)
			f.write(" ".join(item))
			f.write("\n")
		f.write(log)
	with open(cname, 'w') as f:
		for item in candidate:
			item = map(lambda x: str(x), item)
			f.write(" ".join(item))
			f.write("\n")
		f.write(log)



def train_evaluation():
	global data_dict
	datasets = read_input(data_type)  # Change this to change task
	forward_dict, backward_dict = build_indices(datasets[0])
	train_inputs, train_outputs, test_inputs, test_outputs = list(map(lambda x: encode(x, forward_dict), datasets))
	m = model(vocab_size=len(forward_dict), hidden_dim=128)
	optimizer = optim.Adam(m.parameters())
	minibatch_size = 100
	num_minibatches = len(train_inputs) // minibatch_size

	loss_list = []
	timer_list = []

	# A minibatch is a group of examples for which make predictions for and then aggregate before
	# making a gradient update. This helps to make gradient updates more stable
	# as opposed to updating the model for every example (minibatch_size = 1). It also makes better use of the data
	# than performing a single gradient update after looking at the entire dataset (minibatch_size = dataset_size)
	start_training = time.time()
	for epoch in (range(2)):
		# Training
		print("Training")
		# Put the model in training mode
		m.train()
		start_train = time.time()

		for group in tqdm(range(num_minibatches)):
			total_loss = None
			optimizer.zero_grad()
			for i in range(group * minibatch_size, (group + 1) * minibatch_size):
				input_seq = train_inputs[i]
				gold_seq = torch.tensor(train_outputs[i])
				predictions, predicted_seq = m(input_seq, gold_seq)
				loss = m.compute_Loss(predictions, gold_seq)
				# On the first gradient update
				if total_loss is None:
					total_loss = loss
				else:
					total_loss += loss
			timer_list.append(time.time() - start_training)
			loss_list.append(total_loss.data.cpu().numpy())
			total_loss.backward()
			optimizer.step()
		print("Training time: {} for epoch {}".format(time.time() - start_train, epoch))

	if not data_type in data_dict:
		data_dict[data_type] = {}
	if not str(current_value) in data_dict[data_type]:
		data_dict[data_type][str(current_value)] = {}

	data_dict[data_type][str(current_value)]["batch"] = minibatch_size
	data_dict[data_type][str(current_value)]["loss"] = loss_list
	data_dict[data_type][str(current_value)]["timer"] = timer_list

	# Evaluation
	print("Evaluation")
	# Put the model in evaluation mode
	m.eval()
	start_eval = time.time()

	predictions = 0
	correct = 0  # number of tokens predicted correctly
	references = []
	candidates = []
	for input_seq, gold_seq in zip(test_inputs, test_outputs):
		_, predicted_seq = m(input_seq)
		# Hint: why is this true? why is this assumption needed (for now)?
		assert len(predicted_seq) == len(gold_seq)
		correct += sum([predicted_seq[i] == gold_seq[i] for i in range(len(gold_seq))])
		predictions += len(gold_seq)
		# Hint: You might find the following useful for debugging.
		predicted_words = [backward_dict[index] for index in predicted_seq]
		predicted_sentence = " ".join(predicted_words)
		gold_words = [backward_dict[index] for index in gold_seq]
		gold_sentence = " ".join(gold_words)
		candidates.append(predicted_words)
		references.append(gold_words)
	accuracy = correct / predictions
	assert 0 <= accuracy <= 1
	log = "Evaluation time: {} for epoch {}, Accuracy: {}".format(time.time() - start_eval, epoch, accuracy)
	saveLog(references, candidates, log)
	print(log)

def experiment(parameter):
	global max_len, vocab_size, num_examples, varyParameters, data_type, current_value, special_num
	types = ["copy", "reverse", "sort"]

	if parameter == "length":
		varyParameters = "varySequenceLength"
		max_list = [20, 40, 60, 80, 100]
		for length in max_list:
			max_len = length
			current_value = length
			for type in types:
				data_type = type
				command = "./toy.sh %s %s %s %s" % (type, num_examples, vocab_size, max_len)
				print(command)
				os.system(command)
				train_evaluation()
	elif parameter == "dict":
		varyParameters = "varyDictSize"
		vocab_list = [20, 40, 60, 80, 100]
		for size in vocab_list:
			vocab_size = size
			current_value = size
			for type in types:
				data_type = type
				command = "./toy.sh %s %s %s %s" % (type, num_examples, vocab_size, max_len)
				print(command)
				os.system(command)
				train_evaluation()
	elif parameter == "special":
		varyParameters = "varySpecialSize"
		special_List = [0, 1, 2, 3, 4, 5]
		for special in special_List:
			special_num = special
			current_value = special
			for type in types:
				data_type = type
				command = "./toy.sh %s %s %s %s %s" % (type, num_examples, vocab_size, max_len, special_num)
				print(command)
				os.system(command)
				train_evaluation()
	else:
		varyParameters = "varyTrainingSize"
		num_List = [2000, 4000, 6000, 8000, 10000]
		for num in num_List:
			num_examples = num
			current_value = num
			for type in types:
				data_type = type
				command = "./toy.sh %s %s %s %s" % (type, num_examples, vocab_size, max_len)
				print(command)
				os.system(command)
				train_evaluation()
	saveLoss(data_dict)


if __name__ == '__main__':
	# train_evaluation("sort")
	num_examples = 2000
	vocab_size = 20
	max_len = 20
	special_num = 5
	data_type = "sort"
	varyParameters = ""
	current_value = 0
	data_dict = {}

	experiment("train")
