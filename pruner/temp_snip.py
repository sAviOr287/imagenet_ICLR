import copy
import types
import torch.nn.functional as F

import torch
import torch.nn as nn


def GraSP_fetch_data(dataloader, num_classes, samples_per_class):
	datas = [[] for _ in range(num_classes)]
	labels = [[] for _ in range(num_classes)]
	mark = dict()
	dataloader_iter = iter(dataloader)
	while True:
		inputs, targets = next(dataloader_iter)
		for idx in range(inputs.shape[0]):
			x, y = inputs[idx:idx + 1], targets[idx:idx + 1]
			category = y.item()
			if len(datas[category]) == samples_per_class:
				mark[category] = True
				continue
			datas[category].append(x)
			labels[category].append(y)
		if len(mark) == num_classes:
			break

	X, y = torch.cat([torch.cat(_, 0) for _ in datas]), torch.cat([torch.cat(_) for _ in labels]).view(-1)
	return X, y


def count_total_parameters(net):
	total = 0
	for m in net.modules():
		if isinstance(m, (nn.Linear, nn.Conv2d)):
			total += m.weight.numel()
	return total


def count_fc_parameters(net):
	total = 0
	for m in net.modules():
		if isinstance(m, (nn.Linear)):
			total += m.weight.numel()
	return total

def snip_forward_conv2d(self, x):
	return F.conv2d(x, self.weight * self.weight_mask, self.bias,
	                self.stride, self.padding, self.dilation, self.groups)

def snip_forward_linear(self, x):
	return F.linear(x, self.weight * self.weight_mask, self.bias)



def SNIP(net, ratio, train_dataloader, device, num_classes=10, samples_per_class=25, num_iters=1, T=200, reinit=True):
	eps = 1e-10
	keep_ratio = 1 - ratio
	old_net = net

	# Let's create a fresh copy of the network so that we're not worried about
	# affecting the actual training-phase
	net = copy.deepcopy(net)

	for layer in net.modules():
		if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
			layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
			layer.weight.requires_grad = False

		# Override the forward methods:
		if isinstance(layer, nn.Conv2d):
			layer.forward = types.MethodType(snip_forward_conv2d, layer)

		if isinstance(layer, nn.Linear):
			layer.forward = types.MethodType(snip_forward_linear, layer)

	# Let's create a fresh copy of the network so that we're not worried about
	# affecting the actual training-phase
	# net = copy.deepcopy(net)  # .eval()
	net.zero_grad()

	weights = []
	total_parameters = count_total_parameters(net)
	fc_parameters = count_fc_parameters(net)

	for layer in net.modules():
		if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
			if isinstance(layer, nn.Linear) and reinit:
				# TODO write so that i can use different inits
				nn.init.xavier_normal(layer.weight)
			weights.append(layer.weight)

	# for w in weights:
	# 	w.requires_grad_(True)

	# Grab a single batch from the training dataset
	inputs, targets = next(iter(train_dataloader))
	inputs = inputs.to(device)
	targets = targets.to(device)

	# Compute gradients (but don't apply them)
	outputs = net.forward(inputs)
	loss = F.cross_entropy(outputs, targets)
	loss.backward()

	grads = dict()
	old_modules = list(old_net.modules())
	for idx, layer in enumerate(net.modules()):
		if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
			grads[old_modules[idx]] = torch.abs(layer.weight_mask.grad)

	# Gather all scores in a single vector and normalise
	all_scores = torch.cat([torch.flatten(x) for x in grads.values()])
	norm_factor = torch.abs(torch.sum(all_scores)) + eps
	print("** norm factor:", norm_factor)
	all_scores.div_(norm_factor)


	num_params_to_rm = int(len(all_scores) * (1 - keep_ratio))
	if num_params_to_rm == 0:
		acceptable_score = 1e10
	else:
		threshold, _ = torch.topk(all_scores, num_params_to_rm, sorted=True)
		acceptable_score = threshold[-1]
	print('** accept: ', acceptable_score)
	keep_masks = dict()
	for m, g in grads.items():
		keep_masks[m] = ((g / norm_factor) <= acceptable_score).float()

	print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks.values()])))

	return keep_masks
