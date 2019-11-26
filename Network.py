from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from math import floor
import numpy as np
from more_itertools import iterate

"""
Utility function for computing output of convolutions
takes a tuple of (h,w) and returns a tuple of (h,w)
https://pytorch.org/docs/master/nn.html?highlight=conv2d#torch.nn.Conv2d
* padding controls the amount of implicit zero-paddings on both sides for padding number of points for each dimension.
* stride controls the stride for the cross-correlation, a single number or a tuple.
* dilation controls the spacing between the kernel points; also known as the à trous algorithm. 
It is harder to describe, but this link has a nice visualization of what dilation does.
* groups controls the connections between inputs and outputs. in_channels and out_channels must both be divisible by groups

"""
def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    return h, w

class Network(nn.Module):
	def __init__(self):
		super(Network, self).__init__()
		self.name = '2conv2fc'
		# self.layer = None
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)  # kernel_size=5 -> height & width
		self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

		self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
		self.fc2 = nn.Linear(in_features=120, out_features=60)
		self.out = nn.Linear(in_features=60, out_features=10)

	def __repr__(self):
		return super().__repr__()

	def forward(self, t):
		# (1) input layer
		t = t

		# When want to call the forward() method of a nn.Module instance,
		# we call the actual instance instead of calling the forward() method directly.

		# (2) hidden conv layer
		t = self.conv1(t)
		t = F.relu(t)
		t = F.max_pool2d(t, kernel_size=2, stride=2)

		# (3) hidden conv layer
		t = self.conv2(t)
		t = F.relu(t)
		t = F.max_pool2d(t, kernel_size=2, stride=2)

		# (4) hidden linear layer
		# The 4 * 4 is actually the height and width of each of the 12 output channels.
		t = t.reshape(-1, 12 * 4 * 4)
		t = self.fc1(t)
		t = F.relu(t)

		# (5) hidden linear layer
		t = self.fc2(t)
		t = F.relu(t)

		# (6) output layer
		t = self.out(t)
		# t = F.softmax(t, dim=1)

		return t


class Network2(nn.Module):
	def __init__(self):
		super(Network2, self).__init__()
		self.name = '2conv_24_3fc'
		# self.layer = None
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5)  # kernel_size=5 -> height & width
		self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5)

		self.fc1 = nn.Linear(in_features=24 * 4 * 4, out_features=240)
		self.fc2 = nn.Linear(in_features=240, out_features=120)
		self.fc3 = nn.Linear(in_features=120, out_features=60)
		self.out = nn.Linear(in_features=60, out_features=10)

	def __repr__(self):
		return super().__repr__()

	def forward(self, t):
		# (1) input layer
		t = t

		# When want to call the forward() method of a nn.Module instance,
		# we call the actual instance instead of calling the forward() method directly.

		# (2) hidden conv layer
		t = self.conv1(t)
		t = F.relu(t)
		t = F.max_pool2d(t, kernel_size=2, stride=2)

		# (3) hidden conv layer
		t = self.conv2(t)
		t = F.relu(t)
		t = F.max_pool2d(t, kernel_size=2, stride=2)

		# (4) hidden linear layer
		# The 4 * 4 is actually the height and width of each of the 24 output channels.
		t = t.reshape(-1, 24 * 4 * 4)
		t = self.fc1(t)
		t = F.relu(t)

		# (5) hidden linear layer
		t = self.fc2(t)
		t = F.relu(t)

		# (5) hidden linear layer
		t = self.fc3(t)
		t = F.relu(t)

		# (6) output layer
		t = self.out(t)
		# t = F.softmax(t, dim=1)

		return t

class Network3(nn.Module):
	def __init__(self):
		super(Network3, self).__init__()
		self.name = '2conv_48_3fc'
		# self.layer = None
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=24, kernel_size=5)  # kernel_size=5 -> height & width
		self.conv2 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=5)

		self.fc1 = nn.Linear(in_features=48 * 4 * 4, out_features=480)
		self.fc2 = nn.Linear(in_features=480, out_features=180)
		self.fc3 = nn.Linear(in_features=180, out_features=60)
		self.out = nn.Linear(in_features=60, out_features=10)

	def __repr__(self):
		return super().__repr__()

	def forward(self, t):
		# (1) input layer
		t = t

		# When want to call the forward() method of a nn.Module instance,
		# we call the actual instance instead of calling the forward() method directly.

		# (2) hidden conv layer
		t = self.conv1(t)
		t = F.relu(t)
		t = F.max_pool2d(t, kernel_size=2, stride=2)

		# (3) hidden conv layer
		t = self.conv2(t)
		t = F.relu(t)
		t = F.max_pool2d(t, kernel_size=2, stride=2)

		# (4) hidden linear layer
		# The 4 * 4 is actually the height and width of each of the 24 output channels.
		t = t.reshape(-1, 48 * 4 * 4)
		t = self.fc1(t)
		t = F.relu(t)

		# (5) hidden linear layer
		t = self.fc2(t)
		t = F.relu(t)

		# (5) hidden linear layer
		t = self.fc3(t)
		t = F.relu(t)

		# (6) output layer
		t = self.out(t)
		# t = F.softmax(t, dim=1)

		return t

def construct_conv_layers(conv_input_channels, conv_out, conv_ks, dropout, in_size=(28, 28), batch_normalization=True):
	convSequenceArray = []

	after_conv_output_shape = in_size
	for i, (conv_input_channel, conv_out_channel, kernel_size) \
			in enumerate(zip(conv_input_channels, conv_out, conv_ks)):
		convSequenceArray.append(("conv" + str(i),
		                          nn.Conv2d(in_channels=conv_input_channel, out_channels=conv_out_channel,
		                                    kernel_size=kernel_size)))
		if batch_normalization:
			convSequenceArray.append(("batch_norm" + str(i), nn.BatchNorm2d(num_features=conv_out_channel)))
		convSequenceArray.append(("relu" + str(i), nn.ReLU()))

		convSequenceArray.append(("maxpool" + str(i), nn.MaxPool2d(kernel_size=2, stride=2)))
		if dropout != 0.0:
			convSequenceArray.append(("drop" + str(i), nn.Dropout2d(p=dropout)))

		after_conv_output_shape = conv_output_shape(h_w=after_conv_output_shape, kernel_size=kernel_size,
		                                            stride=1, pad=0, dilation=1)#conv
		after_conv_output_shape = conv_output_shape(h_w=after_conv_output_shape, kernel_size=2,
		                                            stride=2, pad=0, dilation=1)#maxpool

	conv_sequence = nn.Sequential(OrderedDict(convSequenceArray))

	return conv_sequence, after_conv_output_shape


def construct_vgg_conv_layers(conv_input_channels, conv_out, conv_ks, dropout, in_size=(28, 28), batch_normalization=True):
	convSequenceArray = []

	after_conv_output_shape = in_size
	group_num = len(conv_out[0])

	for i, (conv_input_channel, conv_out_channel, kernel_size) \
			in enumerate(zip(conv_input_channels, conv_out, conv_ks)):

		for sub_i in range(group_num):
			convSequenceArray.append(("conv_" + str(i) + "_" + str(sub_i),
									  nn.Conv2d(in_channels=conv_input_channel[sub_i], out_channels=conv_out_channel[sub_i],
												kernel_size=kernel_size[sub_i])))
		if batch_normalization:
			convSequenceArray.append(("batch_norm_" + str(i) + "_" + str(sub_i), nn.BatchNorm2d(num_features=conv_out_channel[sub_i])))
		convSequenceArray.append(("relu_" + str(i) + "_" + str(sub_i), nn.ReLU()))

		convSequenceArray.append(("maxpool_" + str(i) + "_" + str(sub_i), nn.MaxPool2d(kernel_size=2, stride=2)))
		if dropout != 0.0:
			convSequenceArray.append(("drop_" + str(i) + "_" + str(sub_i), nn.Dropout2d(p=dropout)))
			after_conv_output_shape = conv_output_shape(h_w=after_conv_output_shape, kernel_size=kernel_size[sub_i],
														stride=1, pad=0, dilation=1)  # conv
			after_conv_output_shape = conv_output_shape(h_w=after_conv_output_shape, kernel_size=kernel_size[sub_i],
														stride=1, pad=0, dilation=1)  # conv
			after_conv_output_shape = conv_output_shape(h_w=after_conv_output_shape, kernel_size=2,
			                                            stride=2, pad=0, dilation=1)  # maxpool

	conv_sequence = nn.Sequential(OrderedDict(convSequenceArray))

	return conv_sequence, after_conv_output_shape

def construct_linear_layers(linear_input_sizes, lin_out, dropout, batch_normalization=True):
	# construct Linear transformations
	linearSequenceArray = []

	for i, (linear_input_size, linear_out_size) \
			in enumerate(zip(linear_input_sizes, lin_out)):
		linearSequenceArray.append(
			("dense" + str(i), nn.Linear(in_features=linear_input_size, out_features=linear_out_size)))
		if batch_normalization:
			linearSequenceArray.append(("batch_norm" + str(i), nn.BatchNorm1d(num_features=linear_out_size)))
		linearSequenceArray.append(("relu" + str(i), nn.ReLU()))

		if dropout != 0.0:
			linearSequenceArray.append(("drop" + str(i), nn.Dropout2d(p=dropout)))
	linearSequence = nn.Sequential(OrderedDict(linearSequenceArray))
	return linearSequence


class MyBasicNetworkBN(nn.Module):

	@staticmethod
	def construct_net(run, use_batch_norm):
		return MyBasicNetworkBN(conv_out=run.conv_out, conv_ks=run.conv_ks,
									   dropout=run.dropout, lin_out=run.lin_out,
					 in_size = run.in_size, out_size=run.out_size, use_batch_norm=use_batch_norm).to(device=run.device)

	def __init__(self, conv_out=[24, 48], conv_ks=[3, 5], dropout=0.2, lin_out=[400, 120, 60],
				 in_size = (28, 28), out_size=10, use_batch_norm = True):
		if len(conv_out) != len(conv_ks):
			raise Exception('channels and kernel_sizes parameters must match!')
		super(MyBasicNetworkBN, self).__init__()
		self.name = 'LeNet'
		self.use_batch_norm = use_batch_norm
		# self.layer = None

		self.conv_out = conv_out
		self.conv_input_channels = [1] + self.conv_out[:-1]
		self.conv_ks = conv_ks
		self.lin_out = lin_out

		self.dropout = dropout
		self.in_size = in_size
		self.use_batch_norm = use_batch_norm

		self.convSequence, after_conv_output_shape = construct_conv_layers(self.conv_input_channels, self.conv_out,
		                                                                   self.conv_ks, self.dropout, self.in_size, self.use_batch_norm)
		in_features_after_conv = np.prod(after_conv_output_shape) * conv_out[-1]
		self.linear_input_sizes = [in_features_after_conv] + lin_out[:-1]

		self.linearSequence = construct_linear_layers(self.linear_input_sizes, self.lin_out, dropout, self.use_batch_norm)

		self.out = nn.Linear(in_features=lin_out[-1], out_features=out_size)

		print("self.conv_out, self.conv_input_channels,  self.conv_ks, self.lin_out, self.linear_input_sizes ",
			  self.conv_out, self.conv_input_channels,  self.conv_ks, self.lin_out, self.linear_input_sizes )

	def __repr__(self):
		return super().__repr__()

	def forward(self, t):
		t = self.convSequence(t)
		t = t.view(t.size(0), -1)
		t = self.linearSequence(t)
		# (6) output layer
		t = self.out(t)

		return t

#Conv-Pool-Conv-Pool with double the filters every stage
class BiggerLeNet(nn.Module):

	@staticmethod
	def construct_net(run, use_batch_norm):
		return BiggerLeNet(conv_out=run.conv_out, conv_ks=run.conv_ks,
		                        dropout=run.dropout, lin_out=run.lin_out,
		                        in_size=run.in_size, out_size=run.out_size,
		                        use_batch_norm=use_batch_norm).to(device=run.device)

	def __init__(self, conv_out=[32, 64, 128], conv_ks=[3, 3, 3], dropout=0.2, lin_out=[400, 120, 60],
				 in_size = (28, 28), out_size=10, use_batch_norm = True):
		if len(conv_out) != len(conv_ks):
			raise Exception('channels and kernel_sizes parameters must match!')
		super(BiggerLeNet, self).__init__()
		self.name = 'BiggerLeNet'
		# self.layer = None

		self.conv_out = conv_out
		self.conv_input_channels = [1] + self.conv_out[:-1]
		self.conv_ks = conv_ks
		self.lin_out = lin_out

		self.dropout = dropout
		self.in_size = in_size
		self.use_batch_norm = use_batch_norm

		self.convSequence, after_conv_output_shape = construct_conv_layers(self.conv_input_channels, self.conv_out,
		                                                                   self.conv_ks, self.dropout, self.in_size, self.use_batch_norm)
		in_features_after_conv = np.prod(after_conv_output_shape) * conv_out[-1]
		self.linear_input_sizes = [in_features_after_conv] + lin_out[:-1]

		self.linearSequence = construct_linear_layers(self.linear_input_sizes, self.lin_out, dropout)

		self.out = nn.Linear(in_features=lin_out[-1], out_features=out_size)

		print("self.conv_out, self.conv_input_channels,  self.conv_ks, self.lin_out, self.linear_input_sizes ",
			  self.conv_out, self.conv_input_channels,  self.conv_ks, self.lin_out, self.linear_input_sizes )

	def __repr__(self):
		return super().__repr__()

	def forward(self, t):
		t = self.convSequence(t)
		t = t.view(t.size(0), -1)
		t = self.linearSequence(t)
		# (6) output layer
		t = self.out(t)

		return t

#Conv-Conv-Pool-Conv-Conv-Pool
class VggLikeNet(nn.Module):

	@staticmethod
	def construct_net(run, use_batch_norm):
		return VggLikeNet(conv_out=run.conv_out, conv_ks=run.conv_ks,
		                   dropout=run.dropout, lin_out=run.lin_out,
		                   in_size=run.in_size, out_size=run.out_size,
		                   use_batch_norm=use_batch_norm).to(device=run.device)

	def __init__(self, conv_out=[[16, 16], [32, 32]], conv_ks=[[3, 3], [3, 3]], dropout=0.2, lin_out=[500],
				 in_size = (28, 28), out_size=10, use_batch_norm = True):
		if len(conv_out) != len(conv_ks):
			raise Exception('channels and kernel_sizes parameters must match!')
		super(VggLikeNet, self).__init__()
		self.name = 'VggLikeNet'

		self.conv_out = conv_out

		conv_out_np = np.array(conv_out)
		conv_out_flat = conv_out_np.flatten()

		conv_input_channels_flat = np.concatenate((np.array([1]), conv_out_flat[:-1]))
		conv_out_np = conv_input_channels_flat.reshape(conv_out_np.shape)
		self.conv_input_channels = conv_out_np.tolist()

		self.conv_ks = conv_ks
		self.lin_out = lin_out

		self.dropout = dropout
		self.in_size = in_size
		self.use_batch_norm = use_batch_norm

		self.convSequence, after_conv_output_shape = construct_vgg_conv_layers(self.conv_input_channels, self.conv_out,
		                                                                   self.conv_ks, self.dropout, self.in_size, self.use_batch_norm)
		in_features_after_conv = np.prod(after_conv_output_shape) * conv_out[-1][-1]
		self.linear_input_sizes = [in_features_after_conv] + lin_out[:-1]

		self.linearSequence = construct_linear_layers(self.linear_input_sizes, self.lin_out, dropout)

		self.out = nn.Linear(in_features=lin_out[-1], out_features=out_size)

		print("self.conv_out, self.conv_input_channels,  self.conv_ks, self.lin_out, self.linear_input_sizes ",
			  self.conv_out, self.conv_input_channels,  self.conv_ks, self.lin_out, self.linear_input_sizes )

	def __repr__(self):
		return super().__repr__()

	def forward(self, t):
		t = self.convSequence(t)
		t = t.view(t.size(0), -1)
		t = self.linearSequence(t)
		# (6) output layer
		t = self.out(t)

		return t