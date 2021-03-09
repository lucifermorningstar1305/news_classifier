import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchtext
import torchtext.legacy.data as td
import argparse
import os
import sys
import time
import wandb

from model import Classifier
from train import *
from utils import *




if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument('--action', '-a', help='whether to train/test the model')
	parser.add_argument('--epochs', '-e', type=int, default=20, help='define the number of epochs')
	parser.add_argument('--learning', '-lr', type=float, default=0.001, help='define the learning rate')
	parser.add_argument('--train_batch_sz', '-tbz', type=int, default=32, help='define the train batch size')
	parser.add_argument('--valid_batch_sz', '-vbz', type=int, default=256, help='define the validation batch size')
	parser.add_argument('--seed', '-s', type=int, default=42, help='define the seed value for pytorch random tensors')

	args = parser.parse_args()

	config = {
			'action': args.action,
			'epochs':args.epochs, 
			'train_batch_sz':args.train_batch_sz,
			'valid_batch_sz':args.valid_batch_sz,
			'learning_rate':args.learning,
			'seed' : args.seed}


	print(f'Configuration Used : {config}')


	
	# Get the classes
	classes = pd.read_csv("../DATA/News/class_list.csv")
	n_classes = classes.shape[0]

	# Define TEXT and LABEL Fields

	TEXT = td.Field(sequential=True, batch_first=True, lower=False, tokenize=str.split, fix_length=10, pad_first=True, include_lengths=True)
	LABEL = td.Field(sequential=False, use_vocab=False, is_target=True)

	train_dataset = td.TabularDataset(path="../../DATA/News/train.csv", format='csv', skip_header=True, fields=[('target', LABEL), ('text', TEXT)])
	valid_dataset = td.TabularDataset(path="../../DATA/News/valid.csv", format='csv', skip_header=True, fields=[('target', LABEL), ('text', TEXT)])

	# Build vocabulary
	TEXT.build_vocab(train_dataset, min_freq=5)

	vocab = TEXT.vocab

	# Save the vocabulary
	vocab_df = pd.DataFrame({"words":list(vocab.stoi.keys()), "index":list(vocab.stoi.values())})
	vocab_df.to_csv("../DATA/News/vocab.csv", index=False)

	print(f'Vocab length: {len(vocab)}')

	# Build the iterator

	train_iter = td.Iterator(train_dataset, batch_size=config['train_batch_sz'], sort_key=lambda x: len(x.text), train=True, shuffle=True)

	valid_iter = td.Iterator(valid_dataset, batch_size=config['valid_batch_sz'], sort_key=lambda x: len(x.text), train=False, shuffle=False)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	model = Classifier(len(vocab), n_classes, 20, hidden_dim=15, drop_val=0.25, device=device)

	model = model.to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

	

	torch.manual_seed(config['seed'])

	if config['action'] == 'train':
		# Initialize wandb for logging metrics
		wandb.init(project='news-classifier', config=config)

		wandb.watch(model, log='all')
		print('Launching training.....')

		train_losses, valid_losses, valid_acc, valid_f1 = train(model, optimizer, 
			train_iter, valid_iter, epochs=config['epochs'], device=device, file_path="../../MODELS/News/")

		plt.title("Epochs vs Losses")
		plt.plot(train_losses, label='Train loss')
		plt.plot(valid_losses, label='Valid Loss')
		plt.xlabel("Epochs")
		plt.ylabel("Losses")
		plt.legend()
		plt.show()


	elif config['action'] == 'test':

		model = Classifier(len(vocab), n_classes, 20, hidden_dim=15, drop_val=0.25, device=torch.device("cpu"))

		optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

		model = model.to(torch.device("cpu"))


		text = input('Enter your text : ')

		model, optimizer = load_chkpt("../../MODELS/News/model.pt", model, optimizer, torch.device("cpu"))

		print("Model Load complete ....")


		text_preprocessed = preprocess(text)

		encoded_text = []

		for words in text_preprocessed.split():

			encoded_text.append(vocab.stoi.get(words, vocab.stoi['<unk>']))

		min_length = min(len(encoded_text), 10)
		encoded_text = encoded_text[:min_length]

		encoded_text = torch.from_numpy(np.asarray(encoded_text, dtype=int))
		encoded_text = encoded_text.unsqueeze(0)
		model.eval()
		y_pred = model(encoded_text, len(encoded_text))
		y_pred = nn.Softmax(dim=1)(y_pred)
		y_pred = torch.argmax(y_pred, dim=1)
		y_pred = y_pred.detach().numpy()

		y_pred = y_pred[0]

		classes = pd.read_csv("../../DATA/News/class_list.csv")

		ans = classes.loc[classes['index'] == y_pred, 'classes'].values[0]

		print(f'This is a {ans} news')