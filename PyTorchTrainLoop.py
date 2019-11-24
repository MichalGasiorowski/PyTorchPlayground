import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

from collections import OrderedDict
from RunBuilder import RunBuilder
from RunManager import RunManager

import pdb

import random
import os

torch.set_printoptions(linewidth=120)

from Network import Network
from Network import Network2
from Network import Network3
from Network import MyBasicNetworkBN

def create_datasets():
	train_transform = transforms.Compose([
		transforms.ToTensor()
	])

	valid_transform = transforms.Compose([
		transforms.ToTensor()
	])

	train_set = torchvision.datasets.FashionMNIST(
		root='.data'
		, train=True
		, download=True
		, transform=train_transform
	)

	valid_set = torchvision.datasets.FashionMNIST(
		root='.data'
		, train=True
		, download=True
		, transform=valid_transform
	)

	test_set = torchvision.datasets.FashionMNIST(
		root='.data'
		, train=False
		, download=True
		, transform=transforms.Compose([
			transforms.ToTensor()
		])
	)

	return (train_set, valid_set, test_set)

def get_train_valid_test_loader(train_set, valid_set, test_set, batch_size, random_seed, valid_split=0.1, shuffle=False, num_workers=4, pin_memory=False):
	num_train = len(train_set)
	indices = list(range(num_train))
	split = int(np.floor(valid_split * num_train))

	if shuffle:
		np.random.seed(random_seed)
		np.random.shuffle(indices)

	train_idx, valid_idx = indices[split:], indices[:split]
	train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
	valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

	train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler,
											   num_workers=num_workers, pin_memory=pin_memory)
	valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, sampler=valid_sampler,
											   num_workers=num_workers, pin_memory=pin_memory)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=shuffle,
											  num_workers=num_workers)

	return (train_loader, valid_loader, test_loader)

def main():

	print(torch.__version__)
	print(torchvision.__version__)

	train_set, valid_set, test_set = create_datasets()

	device = None
	if torch.cuda.is_available():
		device = 'cuda'
	else:
		device = 'cpu'

	networks = { '2conv2fc': lambda: Network(), '2conv_24_3fc': lambda: Network2(), '2conv_48_3fc' : lambda: Network3()}

	'''
	parameters = OrderedDict(
		#network=['2conv_48_3fc']
		network=['2conv2fc', '2conv_24_3fc', '2conv_48_3fc']
		#network=list(networks.keys())
		, lr=[.01, .001]
		, batch_size=[1000]
		, shuffle=[True]
		, epochs=[10]
		, device=[device]
		, nw = [2]
	)
	'''

	parametersLeNet = OrderedDict(
		network=['LeNet']
		, lr=[.005, .01]
		, batch_size=[1000]
		, shuffle=[True]
		, epochs=[3]
		, device=[device]
		, nw=[2]
		, conv_out=[[16, 32], [24, 48], [32, 64]]
		, conv_ks=[[3, 3], [3, 5], [5, 5]]
		, dropout=[0.0, 0.2, 0.5]
		, lin_out=[[200, 84], [500, 200, 84]]
		, in_size=[(28, 28)]
		, out_size=[10]
	)

	'''
	self, conv_out=[24, 48], conv_ks=[3, 5], dropout=0.2, lin_out=[400, 120, 60],
				 in_size = (28, 28), out_size=10):
	'''
	m = RunManager()
	i=0
	for run in RunBuilder.get_runs(parametersLeNet):
		#print(len(RunBuilder.get_runs(parametersLeNet)))
		if i > 3:
			break
		i += 1

		print("Run starting:", run)

		random_seed = random.seed()
		valid_split = 0.1
		pin_memory = (run.device != 'cpu')

		train_loader, valid_loader, test_loader = get_train_valid_test_loader(train_set, valid_set, test_set,
													run.batch_size, random_seed, valid_split,
													run.shuffle, run.nw, pin_memory)
		network = None
		#network = networks.get(run.network)().to(device=run.device)
		if run.network == 'LeNet':
			network = MyBasicNetworkBN(conv_out=run.conv_out, conv_ks=run.conv_ks,
								   dropout=run.dropout, lin_out=run.lin_out,
				 in_size = run.in_size, out_size=run.out_size).to(device=run.device)

		print('network.name: :', network.name, 'chosen')
		print("network architecture: \n", network)

		optimizer = optim.Adam(network.parameters(), lr=run.lr)

		m.begin_run(run, network, run.device, train_loader, valid_loader, test_loader, valid_split, names=train_set.classes)
		for epoch in range(run.epochs):
			m.begin_epoch()
			for batch in train_loader:
				images, labels = batch
				images, labels = images.to(run.device), labels.to(run.device)

				preds = network(images)  # Pass batch

				loss = F.cross_entropy(preds, labels)  # Calculate Loss

				optimizer.zero_grad()  # zero the gradient, tnew gradient is added to old one
				loss.backward()
				optimizer.step()

				m.track_loss(loss)
				m.track_num_correct(preds, labels)

			#m.calculate_test_loss()
			m.calculate_valid_loss()
			m.end_epoch()
		m.calculate_confusion_matrix()
		m.end_run()


	best_models = sorted(m.best_models, key= lambda item: item[0], reverse= True)
	best_models_str = "\n".join(["Valid_accuracy:" + str(item[0]) + "\nHyperParametrs:\n" + "run:\n" + str(item[1][0]) + "\nNetwork:\n" +
	                             str(item[1][1]) for item in best_models[:10]])

	print(best_models_str)

	m.save('results')
	with open('best_models.txt', 'w', encoding='utf-8') as f:
		f.write(best_models_str)


if __name__ == '__main__':
    #main('c:\\Users\\michal\\Google Drive\\Colab Notebooks\\Workspace\\PyTorchPlayground\\')
    main()