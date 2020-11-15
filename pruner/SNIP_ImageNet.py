import copy
import types

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


def GraSP_fetch_data(dataloader, num_classes, samples_per_class):
	print("FECTHING DATA")
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
	print("DONE FETCHING DATA")
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


# def GraSP(net, ratio, train_dataloader, device, num_classes=10, samples_per_class=25, num_iters=1):
# 	eps = 1e-10
# 	keep_ratio = 1 - ratio
# 	old_net = net
#
# 	net = copy.deepcopy(net)
# 	net.zero_grad()
# 	weights = []
# 	total_parameters = count_total_parameters(net)
# 	fc_parameters = count_fc_parameters(net)
#
# 	fc_layers = []
# 	for layer in net.modules():
# 		if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
# 			if isinstance(layer, nn.Linear):
# 				fc_layers.append(layer)
# 			weights.append(layer.weight)
# 	nn.init.xavier_normal(fc_layers[-1].weight)
#
# 	inputs_one = []
# 	targets_one = []
#
# 	grad_w = None
# 	grad_f = None
# 	for w in weights:
# 		w.requires_grad_(True)
#
# 	intvs = {
# 		'cifar10': 128,
# 		'cifar100': 256,
# 		'tiny_imagenet': 128,
# 		'imagenet': 20
# 	}
# 	print_once = False
# 	dataloader_iter = iter(train_dataloader)
# 	for it in range(num_iters):
# 		print("(1): Iterations %d/%d." % (it, num_iters))
# 		inputs, targets = next(dataloader_iter)
# 		N = inputs.shape[0]
# 		din = copy.deepcopy(inputs)
# 		dtarget = copy.deepcopy(targets)
#
# 		start = 0
# 		intv = 20
#
# 		while start < N:
# 			end = min(start + intv, N)
# 			print('(1):  %d -> %d.' % (start, end))
# 			inputs_one.append(din[start:end])
# 			targets_one.append(dtarget[start:end])
# 			outputs = net.forward(inputs[start:end].to(device)) / 200  # divide by temperature to make it uniform
# 			if print_once:
# 				x = F.softmax(outputs)
# 				print(x)
# 				print(x.max(), x.min())
# 				print_once = False
# 			loss = F.cross_entropy(outputs, targets[start:end].to(device))
# 			grad_w_p = autograd.grad(loss, weights, create_graph=False)
# 			if grad_w is None:
# 				grad_w = list(grad_w_p)
# 			else:
# 				for idx in range(len(grad_w)):
# 					grad_w[idx] += grad_w_p[idx]
# 			start = end
#
# 	for it in range(len(inputs_one)):
# 		print("(2): Iterations %d/%d." % (it, len(inputs_one)))
# 		inputs = inputs_one.pop(0).to(device)
# 		targets = targets_one.pop(0).to(device)
# 		outputs = net.forward(inputs) / 200  # divide by temperature to make it uniform
# 		loss = F.cross_entropy(outputs, targets)
# 		grad_f = autograd.grad(loss, weights, create_graph=True)
# 		z = 0
# 		count = 0
# 		for layer in net.modules():
# 			if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
# 				z += (grad_w[count] * grad_f[count]).sum()
# 				count += 1
# 		z.backward()
#
# 	grads = dict()
# 	old_modules = list(old_net.modules())
# 	for idx, layer in enumerate(net.modules()):
# 		if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
# 			grads[old_modules[idx]] = -layer.weight.data * layer.weight.grad  # -theta_q Hg
#
# 	# Gather all scores in a single vector and normalise
# 	all_scores = torch.cat([torch.flatten(x) for x in grads.values()])
# 	norm_factor = torch.abs(torch.sum(all_scores)) + eps
# 	print("** norm factor:", norm_factor)
# 	all_scores.div_(norm_factor)
#
# 	num_params_to_rm = int(len(all_scores) * (1 - keep_ratio))
# 	threshold, _ = torch.topk(all_scores, num_params_to_rm, sorted=True)
# 	# import pdb; pdb.set_trace()
# 	acceptable_score = threshold[-1]
# 	print('** accept: ', acceptable_score)
# 	keep_masks = dict()
# 	for m, g in grads.items():
# 		keep_masks[m] = ((g / norm_factor) <= acceptable_score).float()
#
# 	print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks.values()])))
#
# 	return keep_masks


def snip_forward_conv2d(self, x):
	return F.conv2d(x, self.weight * self.weight_mask, self.bias,
	                self.stride, self.padding, self.dilation, self.groups)


def snip_forward_linear(self, x):
	return F.linear(x, self.weight * self.weight_mask, self.bias)


def SNIP(net, ratio, train_dataloader, device, num_classes=1000, samples_per_class=1, num_iters=1, T=200, reinit=False,
         scaled_init=False):
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

	net.zero_grad()

	# for layer in net.modules():
	# 	if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
	# 		if isinstance(layer, nn.Linear) and reinit:
	# 			# TODO write so that i can use different inits
	# 			nn.init.xavier_normal(layer.weight)

	# Grab a single batch from the training dataset

	# inputs, targets = GraSP_fetch_data(train_dataloader, num_classes, samples_per_class)
	# # import pdb ; pdb.set_trace()
	# # inputs, targets = next(iter(train_dataloader))
	# inputs = inputs.to(device)
	# targets = targets.to(device)
	#
	# # Compute gradients (but don't apply them)
	# outputs = net.forward(inputs)
	# loss = F.cross_entropy(outputs, targets)
	# loss.backward()

	weights = []
	fc_layers = []
	for layer in net.modules():
		if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
			if isinstance(layer, nn.Linear):
				fc_layers.append(layer)
			weights.append(layer.weight)
	#nn.init.xavier_normal(fc_layers[-1].weight)

	inputs_one = []
	targets_one = []

	for w in weights:
		w.requires_grad_(True)

	dataloader_iter = iter(train_dataloader)
	print(num_iters)
	for it in range(num_iters):
		print("(1): Iterations %d/%d." % (it, num_iters))
		inputs, targets = next(dataloader_iter)
		N = inputs.shape[0]
		din = copy.deepcopy(inputs)
		dtarget = copy.deepcopy(targets)

		start = 0
		intv = 20

		while start < N:
			end = min(start + intv, N)
			print('(1):  %d -> %d.' % (start, end))
			inputs_one.append(din[start:end])
			targets_one.append(dtarget[start:end])
			start = end

	for it in range(len(inputs_one)):
		print("(2): Iterations %d/%d." % (it, len(inputs_one)))
		inputs = inputs_one.pop(0).to(device)
		targets = targets_one.pop(0).to(device)
		outputs = net.forward(inputs)
		loss = F.cross_entropy(outputs, targets)
		loss.backward()


	grads = dict()
	old_modules = list(old_net.modules())
	for idx, layer in enumerate(net.modules()):
		if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
			# NOTE: We need the negative infront because of the way we are computing the mask later on
			grads[old_modules[idx]] = -torch.abs(layer.weight_mask.grad)

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
	keep_masks_scaled = dict()

	if scaled_init:
		# TODO we set act always to relu for now!!!!!
		act = 'relu'
		with torch.no_grad():
			print(' WE are using SCALED EOC !!!!')
			# Scaling the weights after pruning
			if act == 'elu':
				sigma_w2 = 1.22459
			elif act == 'relu':
				sigma_w2 = 2.
			elif act == 'tanh':
				sigma_w2 = 1.2981
			else:
				raise NotImplementedError

			scaled_values = dict()
			for idx, layer in enumerate(net.modules()):
				if isinstance(layer, nn.Linear):
					layer_length = layer.weight.shape[1]
					scaling = (layer.weight ** 2 * keep_masks[old_modules[idx]] / sigma_w2 ** 2 * layer_length).mean(
							dim=1)
					scaling_vals = 1. / (torch.sqrt(scaling) + 0.001)
					scaled_values[old_modules[idx]] = scaling_vals.view(1, -1).expand(
							keep_masks[old_modules[idx]].shape[1], -1)
				elif isinstance(layer, nn.Conv2d):
					# print(layer.weight.shape)
					# weights shape [channel_out, channel_in, filter_size, filter_size]
					filter_size = layer.weight.shape[-1]
					number_of_channels = layer.weight.shape[1]
					scaling = (layer.weight ** 2 * keep_masks[
						old_modules[idx]] / sigma_w2 ** 2 * number_of_channels).mean(dim=1)
					scaling_vals = 1. / (torch.sqrt(scaling) + 0.001)
					scaled_values[old_modules[idx]] = scaling_vals.unsqueeze(1).repeat(1, keep_masks[
						old_modules[idx]].shape[1], 1, 1)

			for idx, layer in enumerate(net.modules()):
				if isinstance(layer, nn.Linear):
					keep_masks_scaled[old_modules[idx]] = scaled_values[old_modules[idx]].t() * keep_masks[
						old_modules[idx]]
				elif isinstance(layer, nn.Conv2d):
					keep_masks_scaled[old_modules[idx]] = scaled_values[old_modules[idx]] * keep_masks[
						old_modules[idx]]

	return keep_masks, keep_masks_scaled
