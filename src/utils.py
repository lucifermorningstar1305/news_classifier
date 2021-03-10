import numpy as np
import re
import string
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn



def preprocess(text):

	text = re.sub(r"[^\x00-\x7F]+", " ", text)
	text = re.sub(f'[{re.escape(string.punctuation)}0-9\\r\\t\\n]', " ", text)
	text = text.lower()
	text = text.split(" ")
	text = list(filter(lambda x: x not in ['',' '], text))

	return " ".join(text).strip()


def count_occurences(all_text):

	counts = Counter()

	for text in all_text:

		counts.update(text)

	return counts


def delete_less_frequent_word(data, min_freq=2):

	for words in list(data):

		if data[words] < min_freq:

			del data[words]


	return data


def build_vocab(data):

	vocab2idx = {"":0, "UNK":1}
	vocab = ["", "UNK"]

	for words in list(data):

		vocab2idx[words] = len(vocab)
		vocab.append(words)

	return vocab2idx, vocab


def encode_sentence(text, vocab2idx, N=10):

	encoded = np.zeros(N, dtype=int)
	main_enc = np.array([vocab2idx.get(word, vocab2idx["UNK"]) for word in text])
	length = min(N, len(main_enc))
	encoded[:length] = main_enc[:length]
	encoded = encoded.tolist()
	encoded = [str(e) for e in encoded]

	return f'{" ".join(encoded)}|{length}'



def calculate_metrics(y_true, y_pred):

	y_pred = nn.Softmax(dim=1)(y_pred)
	y_pred = torch.argmax(y_pred, dim=1)

	y_pred = y_pred.detach().cpu().numpy()
	y_true = y_true.detach().cpu().numpy()

	acc = accuracy_score(y_true, y_pred)
	f1 = f1_score(y_true, y_pred, average='weighted')

	return acc, f1



def load_glove_vectors(file_path):

	word_vectors = {}

	with open(file_path, 'r') as f:

		for line in f:

			data = line.split()
			word_vectors[data[0]] = np.array([float(x) for x in data[1:]])

	return word_vectors



def get_embedd_matrix(pretrained_vec, vocab2idx, embedd_size=50):

	vocab_size = len(vocab2idx)

	W = np.zeros((vocab_size, embedd_size), dtype='float32')
	W[0] = np.random.uniform(-0.25, 0.25, embedd_size) # adding a vector for <UNK> token
	W[1] = np.zeros(embedd_size, dtype='float32') # adding a vector for padding
	
	i = 2
	for words in vocab2idx.keys():

		if words not in ['<unk>', '<pad>']:

			W[i] = pretrained_vec.get(words, np.random.uniform(-0.25, 0.25, embedd_size))
			i += 1


	return W




	