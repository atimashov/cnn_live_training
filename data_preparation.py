import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class MyDataset(Dataset):
	def __init__(self, root, model = 'AlexNet', train = True):
		self.root = root
		self.model = model
		self.train = train
		classes = os.listdir(root)
		n_class = 10**10 # just any big number larger than amount of images in each class

		# check number of photos for each class (to be sure we create balanced set)
		for my_class in classes:
			n = len(os.listdir(os.path.join(root, my_class)))
			if n < n_class: n_class = n
		train_val =  int(0.9 * n_class)

		# create images & labels
		self.imgs = []
		self.labels_idx = []
		self.labels_dict = {}
		for i, my_class in enumerate(classes):
			if train:
				self.imgs.extend([os.path.join(root, my_class, img) for img in os.listdir(os.path.join(root, my_class))[:train_val]])
				self.labels_idx.extend([i] * train_val)
			else:
				self.imgs.extend([os.path.join(root, my_class, img) for img in os.listdir(os.path.join(root, my_class))[train_val:n_class]])
				self.labels_idx.extend([i] * (n_class - train_val))
			self.labels_dict[i] = my_class

	def __len__(self):
		return len(self.imgs)

	def __getitem__(self, idx):
		# load images and targets
		img_path = self.imgs[idx]
		img = Image.open(img_path).convert("RGB")
		target = self.labels_idx[idx]

		if self.model == 'AlexNet':
			if self.train:
				trans = transforms.Compose([
					transforms.Resize((256, 256)),
					transforms.RandomCrop(224),
					transforms.RandomHorizontalFlip(0.5),
					transforms.ToTensor(),
					transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
				])
			else:
				trans = transforms.Compose([
					transforms.Resize((224, 224)),
					transforms.ToTensor(),
					transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
				])
		return trans(img), target
