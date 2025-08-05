import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from itertools import chain
from tqdm import tqdm
from adam_atan2_pytorch import AdamAtan2


batch_size = 32
width = 1600
internal_width = 256
device = "mps"
bias = True
torch.utils.backcompat.broadcast_warning.enabled = True

class ConvNet(nn.Module):
	def __init__(self):
		super(ConvNet, self).__init__()
		self.conv1 = nn.Conv2d(4, 32, 3, 1)
		self.conv2 = nn.Conv2d(32, 64, 3, 1)
		self.fc = nn.Linear(64*16*16, internal_width)
	def forward(self, x):
		x = self.conv1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = F.relu(x)
		x = x.flatten(start_dim=1)
		x = self.fc(x)
		return x

class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()

		self.n_embd = internal_width
		self.n_head = 1
		self.dropout = 0.5
		self.c_attn = nn.Linear(internal_width, 3 * internal_width, bias=bias)
		self.fc_out2 = nn.Linear(internal_width, internal_width)
		self.resid_dropout = nn.Dropout(0.5)
		self.c_proj = nn.Linear(internal_width, internal_width, bias=bias)
		self.c_fc = nn.Linear(internal_width, 4 * internal_width, bias=bias)
		self.c_fc2  = nn.Linear(4 * internal_width, internal_width, bias=bias)
		self.gelu = nn.GELU()
		self.fc_dropout = nn.Dropout(0.5)
		self.x_proj = nn.Linear(width, internal_width)

	def forward(self, zL, zH, x=None):

		if len(zL.shape) == 2:
			zL = zL.unsqueeze(dim=1)
		if len(zH.shape) == 2:
			zH = zH.unsqueeze(dim=1)
		if x != None and len(x.shape) == 2:
			x = x.unsqueeze(dim=1)

		if zL.shape[0] > zH.shape[0]:
			zH = zH.repeat(batch_size, 1, 1)
		if zL.shape[0] < zH.shape[0]:
			zL = zL.repeat(batch_size, 1, 1)

		if x == None:
			x = torch.concat([zL, zH], dim=1)
		else:
			if x.shape[0] > zL.shape[0]:
				zL = zL.repeat(batch_size, 1, 1)
			if x.shape[0] > zH.shape[0]:
				zH = zH.repeat(batch_size, 1, 1)
			if x.shape[0] < zL.shape[0]:
				x = x.repeat(batch_size, 1, 1)
			x = torch.concat([zL, zH, x], dim=1)

		# Causal Self-Attention
		B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
		q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
		k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
		q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
		v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
		y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=False)
		y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
		y = self.resid_dropout(self.c_proj(y))

		# MLP
		y = self.c_fc(y)
		y = self.gelu(y)
		y = self.c_fc2(y)
		y = self.fc_dropout(y) + x

		y = y[:,-1:,:]

		return y # zL

class InputHead(nn.Module):
	def __init__(self):
		super(InputHead, self).__init__()
		self.fc1 = nn.Linear(width, internal_width)

	def forward(self, x):
		x = x.flatten(start_dim=1)
		x = self.fc1(x)
		return x

class OutputHead(nn.Module):
	def __init__(self, output_size=4):
		super(OutputHead, self).__init__()
		self.fc1 = nn.Linear(internal_width, output_size)

	def forward(self, x):
		x = x.flatten(start_dim=1)
		x = self.fc1(x)
		x = F.softmax(x, dim=1)
		return x

class HRM(nn.Module):
	def __init__(self, output_size=4):
		super(HRM, self).__init__()
		self.L_net = Net()
		self.H_net = Net()
		self.output_head = OutputHead(output_size)
		self.input_embedding = ConvNet()
	def forward(self, z, x, N=2, T=2):
		x = x.to(device)
		x = self.input_embedding(x)
		zH, zL = z
		zH = zH.to(device)
		zL = zL.to(device)
		with torch.no_grad():
			for _i in range(N * T - 1):
				zL = self.L_net(zL, zH, x)
				if (_i + 1) % T == 0:
					zH = self.H_net(zH, zL)

		# 1−step grad
		zL = self.L_net(zL, zH, x)
		zH = self.H_net(zH, zL, x)
		return (zH, zL), self.output_head(zH)

z_init_policy = torch.randn((1, internal_width)), torch.randn((1, internal_width)).to(device)
z_init_future = torch.randn((1, internal_width)), torch.randn((1, internal_width)).to(device)

if __name__ == "__main__":

	# Deep Supervision
	y_hats = []
	y_trues = []
	xs = []

	for _ in tqdm(range(1000)):
		x = torch.rand((batch_size, 1)) * 2 * torch.pi 
		y_true = torch.sin(x)
		N_supervision = 100
		z = z_init
		opt = optim.AdamW(chain(L_net.parameters(), H_net.parameters(), input_embedding.parameters(), output_head.parameters()), lr=1e-2)
		for step in range(N_supervision):
			z, y_hat = my_hrm(z, x)
			loss = F.mse_loss(y_hat, y_true)
			z[0].detach()
			z[1].detach()
			loss.backward()
			opt.step()
			opt.zero_grad()
		y_hats.append(y_hat.detach())
		y_trues.append(y_true.detach())
		xs.append(x.detach())
	import matplotlib.pyplot as plt
	print(len(xs))
	print(len(y_hats))
	print(len(y_trues))
	plt.scatter(xs, y_hats, s=0.1)
	plt.scatter(xs, y_trues, s=0.1)
	plt.savefig("plot.png")

