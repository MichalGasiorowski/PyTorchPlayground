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
	if valid_split > 0.0:
		split = int(np.floor(valid_split * num_train))

	if shuffle:
		np.random.seed(random_seed)
		np.random.shuffle(indices)

	train_idx, valid_idx = indices[split:], indices[:split]
	train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
	valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)
	# torch.utils.data.SequentialSampler
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler,
											   num_workers=num_workers, pin_memory=pin_memory)
	valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, sampler=valid_sampler,
											   num_workers=num_workers, pin_memory=pin_memory) if split > 0 else None
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=shuffle,
											  num_workers=num_workers)

	return (train_loader, valid_loader, test_loader)

def get_train_valid_test_loader_for_final_training(train_set, valid_set, test_set, batch_size, random_seed,
								shuffle=False, num_workers=4, pin_memory=False):
	train_loader, valid_loader, test_loader = get_train_valid_test_loader(train_set=train_set, valid_set=valid_set, test_set=test_set,
																		  batch_size=batch_size, random_seed=random_seed, valid_split=0.0,
																		  shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

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
		, epochs=[10]
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
		, epochs=[10]
		, device=[device]
		, nw=[2]
		, conv_out=[[16, 32, 64], [32, 64, 128], [48, 96, 192]]
		, conv_ks=[[3, 3, 3]]
		, dropout=[0.0, 0.2, 0.5]
		, lin_out=[[200, 84], [512], [200]]
		, in_size=[(28, 28)]
		, out_size=[10]
		#, batch_norm=[True]
	)

	parametersVggLikeNet = OrderedDict(
		network=['VggLikeNet']
		, lr=[.001, .005, .01]
		, batch_size=[1000]
		, epochs=[10]
		, device=[device]
		, nw=[2]
		, conv_out=[[[16, 16], [32, 32]], [[24, 24], [48, 48]], [[32, 32], [64, 64]]]
		, conv_ks=[[[3, 3], [3, 3]]]
		, dropout=[0.0, 0.2, 0.5]
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
		for run in RunBuilder.get_runs(parameters):

			print("Run starting:", run)

			random_seed = random.seed()
			valid_split = 0.1
			pin_memory = (run.device != 'cpu')

			train_loader, valid_loader, test_loader = get_train_valid_test_loader(train_set, valid_set, test_set,
														run.batch_size, random_seed, valid_split,
														True, run.nw, pin_memory)

			network = construct_network(run, use_batch_norm)

			print('network.name: :', networkName, ' chosen')
			print("network architecture: \n", network)

			optimizer = optim.Adam(network.parameters(), lr=run.lr)

			runManager_train(runManager=m, run=run, network=network, optimizer=optimizer,
							 train_loader=train_loader, valid_loader=valid_loader, test_loader=None,
							 valid_split=valid_split, names=train_set.classes)

			best_models = sorted(m.best_models, reverse=True)

			best_models_str = "\n".join(str(model) for model in best_models[:5])

			runs_data[networkName] = best_models
			m.save(f'results_{networkName}')
			with open(f'best_models_{networkName}.txt', 'w', encoding='utf-8') as f:
				f.write(best_models_str)

	return runs_data

def runManager_final_train(runManager, run, network, optimizer, train_loader, test_loader, names):

	runManager.begin_run(run=run, network=network, device=run.device,
						 train_loader=train_loader, valid_loader=None, test_loader=test_loader,
						 valid_split = 0.0, names=names, final_train = True)
	for epoch in range(run.epochs):
		runManager.begin_epoch()
		for batch in train_loader:
			images, labels = batch
			images, labels = images.to(run.device), labels.to(run.device)

			preds = network(images)  # Pass batch

			loss = F.cross_entropy(preds, labels)  # Calculate Loss

			optimizer.zero_grad()  # zero the gradient, tnew gradient is added to old one
			loss.backward()
			optimizer.step()

		runManager.calculate_test_loss()
		runManager.end_epoch()

	runManager.calculate_confusion_matrix(runManager.test_loader)
	runManager.end_run()

def runManager_train(runManager, run, network, optimizer, train_loader, valid_loader, test_loader, valid_split, names):

	runManager.begin_run(run, network, run.device, train_loader, valid_loader, test_loader, valid_split, names=names)

	for epoch in range(run.epochs):
		runManager.begin_epoch()
		for batch in train_loader:
			images, labels = batch
			images, labels = images.to(run.device), labels.to(run.device)

			preds = network(images)  # Pass batch

			loss = F.cross_entropy(preds, labels)  # Calculate Loss

			optimizer.zero_grad()  # zero the gradient, new gradient is added to old one
			loss.backward()
			optimizer.step()

			runManager.track_loss(loss)
			runManager.track_num_correct(preds, labels)

		#runManager.calculate_test_loss()
		runManager.calculate_valid_loss()
		runManager.end_epoch()

	runManager.calculate_confusion_matrix()
	runManager.end_run()


def construct_network(run, use_batch_norm=True):
	if run.network == 'LeNet':
		network = MyBasicNetworkBN(conv_out=run.conv_out, conv_ks=run.conv_ks, dropout=run.dropout, lin_out=run.lin_out,
								   in_size=run.in_size, out_size=run.out_size, use_batch_norm=use_batch_norm).to(device=run.device)
	if run.network == 'BiggerLeNet':
		network = BiggerLeNet(conv_out=run.conv_out, conv_ks=run.conv_ks, dropout=run.dropout, lin_out=run.lin_out,
								   in_size=run.in_size, out_size=run.out_size, use_batch_norm=use_batch_norm).to(device=run.device)
	if run.network == 'VggLikeNet':
		network = VggLikeNet(conv_out=run.conv_out, conv_ks=run.conv_ks, dropout=run.dropout, lin_out=run.lin_out,
								   in_size=run.in_size, out_size=run.out_size, use_batch_norm=use_batch_norm).to(device=run.device)
	return network

if __name__ == '__main__':

	runs_data = main()
	#[(key, str(value[0])) for key, value in runs_data.items()]
	print("runs_data:\n", [(key, best_models) for key, best_models in runs_data.items()])
	train_set, valid_set, test_set = create_datasets()
	random_seed = random.seed()

	for key, value in runs_data.items():
		#best_vgg = runs_data['VggLikeNet'][0]
		best_model = value[0]

		best_run_params = best_model.run_params
		pin_memory = (best_run_params.device != 'cpu')

		train_loader, valid_loader, test_loader = get_train_valid_test_loader_for_final_training(
			train_set, valid_set, test_set, best_run_params.batch_size, random_seed, True, best_run_params.nw, pin_memory)

		best_network = construct_network(best_run_params)
		optimizer = optim.Adam(best_network.parameters(), lr=best_run_params.lr)
		runManager = RunManager()

		runManager_final_train(runManager=runManager, run=best_run_params, network=best_network, optimizer=optimizer,
							   train_loader=train_loader, test_loader=test_loader, names=train_set.classes)

		best_final_models = sorted(runManager.best_models, reverse=True)
		print("After final training:\n", best_final_models[0].run_params.network, "\n")

		best_models_str = str(best_final_models[0])

		print(best_models_str)
