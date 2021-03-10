import numpy as np
import torch 
import torch.nn as nn
import wandb
from tqdm import tqdm
import os

from utils import *



def save_chkpt(model, optimizer, avg_val_loss, avg_val_acc, avg_val_f1, file_path):

	if file_path == None:
		return

	state_dict = {"model" : model.state_dict(),
				"optimizer":optimizer.state_dict(),
				"avg_val_loss":avg_val_loss,
				"avg_val_acc":avg_val_acc,
				"avg_val_f1":avg_val_f1}


	torch.save(state_dict, os.path.join(file_path, 'model.pt'))

	print(f'Model saved to ==> {file_path}')




def load_chkpt(file_path, model, optimizer, device):

	if file_path == None:
		return 

	print(f'Loading model from ==> {file_path}')

	state_dict = torch.load(file_path, map_location=device)

	print(f'Valid Loss : {state_dict["avg_val_loss"]}')
	print(f'Valid Acc : {state_dict["avg_val_acc"]}')
	print(f'Valid F1 : {state_dict["avg_val_f1"]}')


	model.load_state_dict(state_dict['model'])
	optimizer.load_state_dict(state_dict['optimizer'])


	return model, optimizer




def train(model, optimizer, train_iter, val_iter, criterion=nn.CrossEntropyLoss(), epochs=20, 
	best_valid_loss = np.float("Inf"), device="cpu", file_path = None):

	torch.cuda.empty_cache()

	train_losses = []
	valid_losses = []

	valid_acc = []
	valid_f1 = []


	for epoch in range(epochs):

		tloss = []
		vloss = []
		vacc = []
		vf1 = []

		model.train()

		for data in tqdm(train_iter):

			inputs, inputs_len = data.text
			targets = data.target
			inputs = inputs.to(device)
			targets = targets.to(device)

			optimizer.zero_grad()
			y_pred = model(inputs, inputs_len)

			loss = criterion(y_pred, targets)

			loss.backward()
			optimizer.step()


			tloss.append(loss.item())


		model.eval()

		for data in tqdm(val_iter):

			with torch.no_grad():

				inputs, inputs_len = data.text
				targets = data.target

				inputs = inputs.to(device)
				targets = targets.to(device)

				y_pred = model(inputs, inputs_len)

				loss = criterion(y_pred, targets)

				vloss.append(loss.item())

				acc, f1 = calculate_metrics(targets, y_pred)

				vacc.append(acc)
				vf1.append(f1)

		tloss = np.mean(tloss)
		vloss = np.mean(vloss)
		vacc = np.mean(vacc)
		vf1 = np.mean(vf1)

		if best_valid_loss > vloss:
			best_valid_loss = vloss
			save_chkpt(model, optimizer, vloss, vacc, vf1, file_path)


		print(f'\nEpoch : {epoch + 1} / {epochs} | Train loss : {tloss:.4f} | Val loss : {vloss:.4f}, Val Acc : {vacc:.2f}, Val F1-Score : {vf1:.2f}')

		train_losses.append(tloss)
		valid_losses.append(vloss)

		valid_f1.append(vf1)
		valid_acc.append(vacc)

		# wandb.log({
		# 	'val_loss':vloss,
		# 	'val_acc':vacc,
		# 	'val_f1':vf1
		# 	})


	return train_losses, valid_losses, valid_acc, valid_f1




		




