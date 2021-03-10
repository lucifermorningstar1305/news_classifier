import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import sys
from datetime import datetime

from src.utils import preprocess, load_glove_vectors, get_embedd_matrix
from src.model import Classifier


def load_model(model, file_path=None, device=torch.device("cpu")):

	if file_path == None:
		return


	print(f'Loading model from ==> {file_path}')

	state_dict = torch.load(file_path, map_location=device)

	print(f'Validation loss for the model : {state_dict["avg_val_loss"]}')
	print(f'Validation Accuracy of the model : {state_dict["avg_val_acc"]}')
	print(f'Validation F1-Score of the model : {state_dict["avg_val_f1"]}')


	model.load_state_dict(state_dict['model'])


	return model



def build_vocab_class():

	df_vocab = pd.read_csv("./docs/vocab.csv")
	df_classes = pd.read_csv("./docs/class_list.csv")

	vocabulary = dict()
	classes = dict()

	for words, index in df_vocab.values:

		vocabulary[words] = index


	for class_, index in df_classes.values:

		classes[index] = class_


	return vocabulary, classes


def encode_sentence(text, vocab, min_length=10):

	encoded_sent = list()

	if len(text) < min_length:
		text_len = len(text)

		for i in range(min_length - text_len):
			text.insert(i, '<pad>')

	else:
		text = text[:min_length]

	for words in text:

		encoded_sent.append(vocab.get(words, vocab['<unk>']))

	return encoded_sent






def predict(text):

	text = preprocess(text)
	text = text.split()

	print(text)

	vocab, classes = build_vocab_class()
	
	vocab_vec = None

	if not os.path.exists("./model/vocab_vector.100d.pt"): # If there are no model for GLoVE
		glove_vectors = load_glove_vectors("./model/glove.6B.100d.txt")
		vocab_vec = torch.from_numpy(get_embedd_matrix(glove_vectors, vocab, embedd_size=100))
		torch.save(vocab_vec, './model/vocab_vector.100d.pt')

	else:
		vocab_vec = torch.load('./model/vocab_vector.100d.pt')


	encoded_sent = encode_sentence(text, vocab, min_length=30)

	encoded_sent = torch.from_numpy(np.asarray(encoded_sent)) # Convert to Pytorch Tensor

	encoded_sent = encoded_sent.unsqueeze(0) # Add batch dimension

	device = torch.device("cpu")

	model = Classifier(len(vocab), len(classes), 100, hidden_dim=128, embedd_vec = vocab_vec, drop_val=0.5, bidirectional=True, is_trainable=False, device=device) # Initialize the model

	model = load_model(model, file_path="./model/model.pt") # Load the model

	# Compute the encoded sent's prediction
	model.eval()

	pred = model(encoded_sent, len(encoded_sent))
	pred = nn.Softmax(dim=1)(pred)
	prob, _ = torch.max(pred, dim=1)
	pred = torch.argmax(pred, dim=1)
	pred = pred.detach().numpy()
	prob = prob.detach().numpy()
	pred = pred[0]
	prob = prob[0]

	return classes[pred], str(prob)





