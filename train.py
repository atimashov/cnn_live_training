from torch.nn.init import kaiming_normal_
import torch
from torch import optim
from torchvision.models import alexnet, vgg16
from data_preparation import MyDataset
from models import MyAlexNet, MyVGG16
import os
from torch.utils.data import DataLoader
import numpy as np
from time import time, sleep
from datetime import datetime
from termcolor import colored
from helper import get_params, update_db
from argparse import  ArgumentParser

def init_weights(model):
	for layer in model.features:
		if type(layer) in [torch.nn.Conv2d, torch.nn.Linear]:
			kaiming_normal_(layer.weight)
	for layer in model.classifier:
		if type(layer) in [torch.nn.Conv2d, torch.nn.Linear]:
			kaiming_normal_(layer.weight)
	return model

def init_model(inputs):
	# run model
	n_classes = len(os.listdir('{}/{}'.format(os.getcwd(), inputs.dataset_path)))
	if inputs.model == 'my_alexnet':
		model = MyAlexNet(num_classes = n_classes)
	elif inputs.model == 'my_vgg16':
		model = MyVGG16(num_classes=n_classes)
	elif inputs.model == 'alexnet':
		model = alexnet(pretrained=True)  # params.transfer
		if inputs.transfer:
			for param in model.parameters():
				param.requires_grad = False
			num_ftrs = model.classifier[6].in_features
			model.classifier[6] = torch.nn.Linear(num_ftrs, n_classes)
	else:
		model = vgg16(pretrained=True)  # params.transfer
		if inputs.transfer:
			for param in model.parameters():
				param.requires_grad = False
			num_ftrs = model.classifier[6].in_features
			model.classifier[6] = torch.nn.Linear(num_ftrs, n_classes)
	model = model.to(device=device)
	if not inputs.transfer:
		model = init_weights(model)
	return model, n_classes

def get_maps_distr(activations):
	maps = []
	for key in activations:
		# print(key, activations[key].size())
		nn_part, layer_type, number = key.split(' | ')
		row = "('{}', '{}', {},  TIMESTAMP '{}'".format(nn_part, layer_type, number, datetime.now())
		weights = activations[key].flatten().cpu().numpy()

		bins = max(n_classes, min(weights.shape[0] // (inputs.batch_size * 2), 30))
		if (weights < 0).sum() == 0:
			q = np.quantile(weights, 0.95)
			hist, bin_edges = np.histogram(weights[weights <= q], bins=bins)
			hist[-1] += (weights > q).sum()
		else:
			hist, bin_edges = np.histogram(weights, bins=bins)
			hist, bin_edges = '{' + ', '.join(map(str, hist)) + '}', '{' + ', '.join(map(str, bin_edges)) + '}'
		maps.append(row + ", '{}', '{}')".format(bin_edges, hist))
	return maps

def get_accuracy(loader, model, device, loss_func = torch.nn.CrossEntropyLoss()):
	num_correct = 0
	num_samples = 0
	model.eval()  # set model to evaluation mode
	losses = []
	with torch.no_grad():
		for (imgs, labels) in loader:
			imgs = imgs.to(device = device, dtype = dtype)  # move to device, e.g. GPU
			labels = labels.to(device = device, dtype = torch.long)

			scores = model(imgs)
			loss = loss_func(scores, labels)
			losses.append(float(loss))

			_, preds = scores.max(1)
			num_correct += (preds == labels).sum()
			num_samples += preds.size(0)

		acc = 100 * float(num_correct) / num_samples
		loss = sum(losses) / len(losses)
	return acc, loss

def get_optimizer(old_params, model, optimizer = None):
	prev_lr, prev_wd, prev_do, prev_opt, flag = old_params
	if optimizer is None:
		if prev_do != 50:
			for i, layer in enumerate(model.features):
				if type(layer) == torch.nn.Dropout:
					model.features[i] = torch.nn.Dropout(prev_do / 100)
			for i, layer in enumerate(model.classifier):
				if type(layer) == torch.nn.Dropout:
					model.classifier[i] = torch.nn.Dropout(prev_do / 100)
		if prev_opt == 'Adam':
			optimizer = optim.Adam(model.parameters(), lr = prev_lr, weight_decay = prev_wd)
		else:
			optimizer = optim.SGD(model.parameters(), lr = prev_lr, weight_decay = prev_wd, momentum = 0.9, nesterov = True)
	else:
		lr, wd, do, opt, flag = get_params()
		if flag:
			return (lr, wd, do, opt, flag), model, optimizer

		if (lr != prev_lr) or (wd != prev_wd) or (do != prev_do) or (opt != prev_opt):
			if prev_do != do:
				for i, layer in enumerate(model.features):
					if type(layer) == torch.nn.Dropout:
						model.features[i] = torch.nn.Dropout(prev_do / 100)
				for i, layer in enumerate(model.classifier):
					if type(layer) == torch.nn.Dropout:
						model.classifier[i] = torch.nn.Dropout(prev_do / 100)
			prev_lr, prev_wd, prev_do, prev_opt = lr, wd, do, opt

			if prev_opt == 'Adam':
				optimizer = optim.Adam(model.parameters(), lr=prev_lr, weight_decay=prev_wd)
			else:
				optimizer = optim.SGD(model.parameters(), lr=prev_lr, weight_decay=prev_wd, momentum=0.9, nesterov=True)
	return (prev_lr, prev_wd, prev_do, prev_opt, flag), model, optimizer

def train_my(loader, model, dt_start, epochs = 3, params = None, device = None, loss_func = torch.nn.CrossEntropyLoss(), n_print = 50):
	# init optimizer
	print(params)
	params, model, optimizer = get_optimizer(params, model, None)

	step = 0
	for epoch in range(epochs):
		t = time()
		print(colored('-' * 50, 'cyan'))
		print(colored('{} Epoch {}{} {}'.format('-' * 20, ' ' * (2 - len(str(epoch))), epoch, '-' * 20), 'cyan'))
		print(colored('-' * 50, 'cyan'))

		tacc, vacc = 0, 0
		tloss, vloss = 0, 0
		num_samples = 0

		steps = 0
		for idx, (imgs, labels) in enumerate(loader['train']):
			model.train()  # put model to training mode
			imgs = imgs.to(device = device, dtype = dtype)
			labels = labels.to(device = device, dtype = torch.long)


			# get activation maps hooks
			if step % n_print == n_print - 1:
				activations = {}
				def get_activation(name):
					def hook(model, input, output):
						activations[name] = output.detach()
					return hook
				layer_n = 1
				for i, layer in enumerate(model.features):
					if type(layer) not in [torch.nn.Conv2d, torch.nn.Linear]: continue
					name = 'features | {} | {}'.format('conv' if type(layer) == torch.nn.Conv2d else 'fc', layer_n)
					model.features[i].register_forward_hook(get_activation(name))
					layer_n += 1
				layer_n = 1
				for i, layer in enumerate(model.classifier):
					if type(layer) not in [torch.nn.Conv2d, torch.nn.Linear]: continue
					name = 'classifier | {} | {}'.format('conv' if type(layer) == torch.nn.Conv2d else 'fc', layer_n)
					model.classifier[i].register_forward_hook(get_activation(name))
					layer_n += 1

			scores = model(imgs)
			loss = loss_func(scores, labels)

			# Zero out all of the gradients for the variables which the optimizer will update.
			optimizer.zero_grad()

			# Backwards pass and computing gradients
			loss.backward()
			optimizer.step()

			# create checkpoint
			if step % n_print == n_print - 1:
				steps += 1
				# activation maps
				maps = get_maps_distr(activations)
				# train
				_, preds = scores.max(1)
				tacc += torch.sum(preds == labels.data)
				num_samples += preds.size(0)
				tloss += loss.item()
				# validate
				temp_acc, temp_loss = get_accuracy(loader['val'], model, device = device, loss_func = loss_func)
				vacc += temp_acc
				vloss += temp_loss
				# save current step to SQL
				update_db(params, dt_start, epoch, step, num_samples, steps, tloss, tacc, vloss, vacc, maps)
				# display
				print('Iteration {}, loss = {}'.format(idx, round(loss.item(), 2)))
				print('Loss: train = {}, validate = {}'.format(round(tloss / steps, 4), round(vloss / steps, 4)))
				print('Accuracy: train = {}, validate = {}'.format(round(100 * float(tacc) / num_samples, 2), round(vacc / steps, 2)))
				print()


			# check parameters update
			params, model, optimizer = get_optimizer(params, model, optimizer)
			step += 1
			_,_,_,_,flag = params
			if flag: break
		if flag: break

		t = int(time() - t)
		t_min, t_sec = str(t // 60), str(t % 60)
		print(colored('It took {}{} min. {}{} sec.'.format(' ' * (2 - len(t_min)), t_min, ' ' * (2 - len(t_sec)), t_sec), 'cyan'))
		print(colored('-' * 50, 'cyan'))
		print()

if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('--model', type=str, default='my_alexnet', help='model name: my_alexnet, alexnet, my_vgg16, vgg16')
	parser.add_argument('--dataset-path', type=str, default='data', help='path to dataset: image_net_10, corrosion_dataset')
	parser.add_argument('--n-print', type=int, default=50, help='how often to print')
	parser.add_argument('--n-epochs', type=int, default=1000, help='number of epochs')
	parser.add_argument('--batch-size', type=int, default=32, help='batch size')
	parser.add_argument('--transfer', type=str, default='False', help='transfer/full learning')
	parser.add_argument('--use-gpu', type=str, default='True', help='gpu/cpu')
	inputs = parser.parse_args()
	print(inputs)

	inputs.transfer = True if inputs.transfer == 'True' else False
	USE_GPU = True if inputs.use_gpu == 'True' else False
	dtype = torch.float32  # TODO: find out how it affects speed and accuracy
	device = torch.device('cuda:0' if USE_GPU and torch.cuda.is_available() else 'cpu')

	# run model
	model, n_classes = init_model(inputs)
	# waiting for the new input
	while True:
		params = get_params(start = True)
		if params is not None: break
		sleep(10)
	# create data loader
	data_train = MyDataset(root = '{}/{}'.format(os.getcwd(), inputs.dataset_path), train = True)
	data_val = MyDataset(root = '{}/{}'.format(os.getcwd(), inputs.dataset_path), train = False)
	data_loader = {
		'train': DataLoader(data_train, batch_size = inputs.batch_size, shuffle = True, num_workers = 6),
		'val': DataLoader(data_val, batch_size = inputs.batch_size, shuffle = True, num_workers = 6)
	}
	hist = train_my(data_loader, model, datetime.now(), epochs = inputs.n_epochs, params = params, device = device, n_print = inputs.n_print)

