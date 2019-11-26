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
from Network import BiggerLeNet
from Network import VggLikeNet


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

def get_train_valid_test_loader(train_set, valid_set, test_set, batch_size, random_seed, valid_split=0.1,
								shuffle=False, num_workers=4, pin_memory=False):
	num_train = len(train_set)
	indices = list(range(num_train))
	split = 0
	if valid_split >= 0.0:
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
											   num_workers=num_workers, pin_memory=pin_memory) if split > 0 else None
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=shuffle,
											  num_workers=num_workers)

	return (train_loader, valid_loader, test_loader)

def get_train_valid_test_loader_for_final_training(train_set, valid_set, test_set, batch_size, random_seed,
								shuffle=False, num_workers=4, pin_memory=False):
	train_loader, valid_loader, test_loader = get_train_valid_test_loader(train_set, valid_set, test_set,
																		  batch_size, random_seed, 0.0,
																		  shuffle, num_workers, pin_memory)

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
		, epochs=[5]
		, device=[device]
		, nw=[2]
		, conv_out=[[16, 32], [24, 48], [32, 64]]
		, conv_ks=[[3, 3], [3, 5], [5, 5]]
		, dropout=[0.0, 0.2, 0.5]
		, lin_out=[[200, 84], [500, 200, 84]]
		, in_size=[(28, 28)]
		, out_size=[10]
		#, batch_norm=[True]
	)

	parametersBiggerLeNet = OrderedDict(
		network=['BiggerLeNet']
		, lr=[.001, .005, .01]
		, batch_size=[1000]
		, shuffle=[True]
		, epochs=[5]
		, device=[device]
		, nw=[2]
		, conv_out=[[16, 32, 64], [32, 64, 128], [48, 96, 192]]
		, conv_ks=[[3, 3, 3]]
		, dropout=[0.2, 0.5]
		, lin_out=[[200, 84], [512], [200]]
		, in_size=[(28, 28)]
		, out_size=[10]
		#, batch_norm=[True]
	)

	parametersVggLikeNet = OrderedDict(
		network=['VggLikeNet']
		, lr=[.001, .005, .01]
		, batch_size=[1000]
		, shuffle=[True]
		, epochs=[2]
		, device=[device]
		, nw=[2]
		, conv_out=[[[16, 16], [32, 32]], [[24, 24], [48, 48]]]
		, conv_ks=[[[3, 3], [3, 3]]]
		, dropout=[0.2, 0.5]
		, lin_out=[[200, 84], [512], [200]]
		, in_size=[(28, 28)]
		, out_size=[10]
		#, batch_norm=[True]
	)

	runs_data = {}
	best_models = None
	experiments = [('LeNet', parametersLeNet), ('BiggerLeNet', parametersBiggerLeNet), ('VggLikeNet', parametersVggLikeNet)]

	for networkName, parameters in experiments.__reversed__():
		m = RunManager()
		use_batch_norm = True
		i = 0
		for run in RunBuilder.get_runs(parameters):

			#print(len(RunBuilder.get_runs(parametersLeNet)))
			if i > 1:
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
				network = MyBasicNetworkBN(run, use_batch_norm)
			if run.network == 'BiggerLeNet':
				network = BiggerLeNet(run, use_batch_norm)
			if run.network == 'VggLikeNet':
				network = VggLikeNet(run, use_batch_norm)

			print('network.name: :', networkName, ' chosen')
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
			best_models = sorted(m.best_models, key=lambda item: item[0], reverse=True)
			best_models_str = "\n".join(
				["Valid_accuracy:" + str(item[0]) + "\nHyperParameters:\n" + "run:\n" + str(item[1][0]) + "\nNetwork:\n" +
				 str(item[1][1]) for item in best_models[:10]])
			#print(best_models_str)

			runs_data[networkName] = (best_models, best_models_str)
			m.save(f'results_{networkName}')
			with open(f'best_models_{networkName}.txt', 'w', encoding='utf-8') as f:
				f.write(best_models_str)


	return runs_data

if __name__ == '__main__':

	runs_data = main()
	print("runs_data:\n", runs_data)

	#self.best_models.append((self.validCorrect / len(self.valid_loader.sampler), (self.run_params, self.network)))

	best_vgg = runs_data['VggLikeNet'][0][1]

	train_set, valid_set, test_set = create_datasets()
	random_seed = random.seed()

	run = best_vgg[0]
	pin_memory = (run.device != 'cpu')
	train_loader, valid_loader, test_loader = get_train_valid_test_loader_for_final_training(
		train_set, valid_set, test_set, run.batch_size, random_seed, run.shuffle, run.nw, pin_memory)

	best_vgg = VggLikeNet()