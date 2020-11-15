import torch
import torch.nn as nn
import math

def weights_init_kaiming_xavier(m):
	# print('=> weights init')
	if isinstance(m, nn.Conv2d):
		nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
		# nn.init.normal_(m.weight, 0, 0.1)
		if m.bias is not None:
			m.bias.data.zero_()
	elif isinstance(m, nn.Linear):
		# nn.init.xavier_normal(m.weight)
		nn.init.normal_(m.weight, 0, 0.01)
		nn.init.constant_(m.bias, 0)
	elif isinstance(m, nn.BatchNorm2d):
		# Note that BN's running_var/mean are
		# already initialized to 1 and 0 respectively.
		if m.weight is not None:
			m.weight.data.fill_(1.0)
		if m.bias is not None:
			m.bias.data.zero_()

def weights_init_kaiming_relu(m):
	if isinstance(m, nn.Conv2d):
		nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='relu')
		# nn.init.normal_(m.weight, 0, 0.1)
		if m.bias is not None:
			m.bias.data.zero_()
	elif isinstance(m, nn.Linear):
		nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
	elif isinstance(m, nn.BatchNorm2d):
		# Note that BN's running_var/mean are
		# already initialized to 1 and 0 respectively.
		if m.weight is not None:
			m.weight.data.fill_(1.0)
		if m.bias is not None:
			m.bias.data.zero_()

def weights_init_kaiming_tanh(m):
	if isinstance(m, nn.Conv2d):
		nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='tanh')
		# nn.init.normal_(m.weight, 0, 0.1)
		if m.bias is not None:
			m.bias.data.zero_()
	elif isinstance(m, nn.Linear):
		nn.init.kaiming_normal_(m.weight, nonlinearity='tanh')
	elif isinstance(m, nn.BatchNorm2d):
		# Note that BN's running_var/mean are
		# already initialized to 1 and 0 respectively.
		if m.weight is not None:
			m.weight.data.fill_(1.0)
		if m.bias is not None:
			m.bias.data.zero_()


def weights_init_xavier(m):
	if isinstance(m, nn.Conv2d):
		nn.init.xavier_normal_(m.weight)
		if m.bias is not None:
			m.bias.data.zero_()
	elif isinstance(m, nn.Linear):
		nn.init.xavier_normal_(m.weight)
	elif isinstance(m, nn.BatchNorm2d):
		# Note that BN's running_var/mean are
		# already initialized to 1 and 0 respectively.
		if m.weight is not None:
			m.weight.data.fill_(1.0)
		if m.bias is not None:
			m.bias.data.zero_()


def weights_init_EOC(m):
	if isinstance(m, nn.Conv2d):
		EOC_weights(m.weight)
		if m.bias is not None:
			EOC_bias(m.bias)
	elif isinstance(m, nn.Linear):
		EOC_weights(m.weight)
		if m.bias is not None:
			EOC_bias(m.bias)
	elif isinstance(m, nn.BatchNorm2d):
		# Note that BN's running_var/mean are
		# already initialized to 1 and 0 respectively.
		if m.weight is not None:
			m.weight.data.fill_(1.0)
		if m.bias is not None:
			m.bias.data.zero_()
			
			
def weights_init_ord(m):
	if isinstance(m, nn.Conv2d):
		ord_weights(m.weight)
		if m.bias is not None:
			ord_bias(m.bias)
	elif isinstance(m, nn.Linear):
		ord_weights(m.weight)
		if m.bias is not None:
			ord_bias(m.bias)
	elif isinstance(m, nn.BatchNorm2d):
		# Note that BN's running_var/mean are
		# already initialized to 1 and 0 respectively.
		if m.weight is not None:
			m.weight.data.fill_(1.0)
		if m.bias is not None:
			m.bias.data.zero_()


def _calculate_fan_in_and_fan_out(tensor):
	dimensions = tensor.dim()
	if dimensions < 2:
		raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

	if dimensions == 2:  # Linear
		fan_in = tensor.size(1)
		fan_out = tensor.size(0)
	else:
		num_input_fmaps = tensor.size(1)
		num_output_fmaps = tensor.size(0)
		receptive_field_size = 1
		if tensor.dim() > 2:
			receptive_field_size = tensor[0][0].numel()
		fan_in = num_input_fmaps * receptive_field_size
		fan_out = num_output_fmaps * receptive_field_size

	return fan_in, fan_out



def EOC_weights(tensor, act='relu'):
	print('#' * 40)
	print('We are using {} activation on EOC'.format(act))
	print('#' * 40)

	fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
	sigma_w2 = 1.

	if act == 'relu':
		#print('relu')
		sigma_w2 = 2.
		q = 'constant variance'
	elif act == 'tanh':
		#print('tanh')
		sigma_w2 = 1.2981 ** 2
		q = 0.49
	elif act == 'elu':
		#print('elu')
		sigma_w2 = 1.22459 ** 2
		q = 1.01

	std = math.sqrt(sigma_w2 / float(fan_in))
	with torch.no_grad():
		return tensor.normal_(0, std)


def EOC_bias(tensor, act='relu'):
	print('#' * 40)
	print('We are using {} activation on EOC'.format(act))
	print('#' * 40)
	sigma_b2 = 0.

	if act == 'relu':
		sigma_b2 = 1e-16
		q = 'constant variance'
	elif act == 'tanh':
		sigma_b2 = 0.2 ** 2
		q = 0.49
	elif act == 'elu':
		sigma_b2 = 0.2 ** 2
		q = 1.01

	std = math.sqrt(sigma_b2)
	with torch.no_grad():
		return tensor.normal_(0, std)


def ord_weights(tensor,sigma_w2):
	print('#' * 40)
	print('Ordered phase with {}sigma_w2'.format(sigma_w2))
	print('#' * 40)
	fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
	std = math.sqrt(sigma_w2 / float(fan_in))
	with torch.no_grad():
		return tensor.normal_(0, std)


def ord_bias(tensor):
	sigma_b2 = 1.

	std = math.sqrt(sigma_b2)
	with torch.no_grad():
		return tensor.normal_(0, std)
