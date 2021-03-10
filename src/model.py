import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Classifier(nn.Module):

	def __init__(self, vocab_size, n_classes, embedd_dim, n_layers=1, embedd_vec=None, drop_val=0.25,
		hidden_dim=128, is_trainable=True, rnn_module='lstm', bidirectional=False, device=None):

		super(Classifier, self).__init__()

		self.embedd = nn.Embedding(vocab_size, embedd_dim, scale_grad_by_freq=True)
		self.rnn_module = rnn_module
		self.n_layers = n_layers
		self.hidden_dim = hidden_dim
		self.device=device
		
		if embedd_vec is not None:

			self.embedd.weight.data.copy_(embedd_vec)

		if not is_trainable:
			self.embedd.weight.requires_grad=False

		if self.rnn_module == 'lstm':		
			self.lstm = nn.LSTM(embedd_dim, hidden_dim, num_layers=n_layers,  batch_first=True, bidirectional=bidirectional)
		else:
			self.gru = nn.GRU(embedd_dim, hidden_dim, num_layers=n_layers, batch_first=True)

		self.num_direction = 2 if bidirectional else 1
		self.linear = nn.Linear(self.num_direction * hidden_dim, n_classes)
		self.drop = nn.Dropout(p=drop_val)


	def forward(self, X, X_len):

		# Initialize the hidden states
		h0 = torch.zeros(self.n_layers * self.num_direction, X.size(0), self.hidden_dim).to(self.device)
		c0 = torch.zeros(self.n_layers * self.num_direction, X.size(0), self.hidden_dim).to(self.device)


		X = self.embedd(X)

		# X_pack = nn.utils.rnn.pack_padded_sequence(X, X_len, batch_first=True, enforce_sorted=False)
		
		out = None
		
		if self.rnn_module == 'lstm':
			# X_pack = pack_padded_sequence(X, X_len, batch_first=True, enforce_sorted=False)
			lstm_out, (ht, ct) = self.lstm(X, (h0, c0))
			# lstm_out, lstm_out_len = pad_packed_sequence(lstm_out, batch_first=True)
			out, _ = torch.max(lstm_out, 1)

		else:

			gru_out, ht = self.gru(X, h0)
			out, _ = torch.max(gru_out, 1)
			
		# out = ht[-1]
		# out = self.linear1(out)
		# out = self.tanh(out)
		out = self.drop(out)
		out = self.linear(out)
		return out