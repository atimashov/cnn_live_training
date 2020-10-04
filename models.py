from torch import nn
class MyAlexNet(nn.Module):
	def __init__(self, num_classes = 10):
		super(MyAlexNet, self).__init__()
		self.features = nn.Sequential()
		self.classifier = nn.Sequential()
		# layer 1
		self.features.add_module('1_conv_1', nn.Conv2d(3, 96, kernel_size = 11, stride = 4, padding = 2, bias = False))
		self.features.add_module('1_batchnorm', nn.BatchNorm2d(96))
		self.features.add_module('1_relu', nn.ReLU(inplace = True))
		self.features.add_module('1_maxpool', nn.MaxPool2d(kernel_size=3, stride=2))
		# layer 2
		self.features.add_module('2_conv_2', nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, bias=False))
		self.features.add_module('2_batchnorm', nn.BatchNorm2d(256))
		self.features.add_module('2_relu', nn.ReLU(inplace = True))
		self.features.add_module('2_maxpool', nn.MaxPool2d(kernel_size=3, stride=2))
		# layer 3
		self.features.add_module('3_conv_3', nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1))
		self.features.add_module('3_relu', nn.ReLU(inplace=True))
		# layer 4
		self.features.add_module('4_conv_4', nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1))
		self.features.add_module('4_relu', nn.ReLU(inplace=True))
		# layer 5
		self.features.add_module('5_conv_5', nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1))
		self.features.add_module('5_relu', nn.ReLU(inplace=True))
		self.features.add_module('5_maxpool', nn.MaxPool2d(kernel_size=3, stride=2))
		# layer 6
		self.classifier.add_module('6_dropout', nn.Dropout())
		self.classifier.add_module('6_fc_1', nn.Linear(256 * 6 * 6, 4096))
		self.classifier.add_module('6_relu', nn.ReLU(inplace=True))
		# layer 7
		self.classifier.add_module('7_dropout', nn.Dropout())
		self.classifier.add_module('7_fc_2', nn.Linear(4096, 4096))
		self.classifier.add_module('7_relu', nn.ReLU(inplace=True))
		# layer 8
		self.classifier.add_module('8_fc_3', nn.Linear(4096, num_classes))

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), 256 * 6 * 6) # resize for fc layers
		x = self.classifier(x)
		return x


class MyVGG16(nn.Module):
	def __init__(self, num_classes=10):
		super().__init__()
		self.features = nn.Sequential()
		# conv_1_1
		self.features.add_module('1_1_conv_1', nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False))
		self.features.add_module('1_1_relu_1', nn.ReLU(inplace=True))
		self.features.add_module('1_1_conv_2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
		self.features.add_module('1_1_relu_2', nn.ReLU(inplace=True))
		self.features.add_module('1_1_maxpool', nn.MaxPool2d(kernel_size=2, stride=2))
		# conv_1_2
		self.features.add_module('1_2_conv_1', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False))
		self.features.add_module('1_2_relu_1', nn.ReLU(inplace=True))
		self.features.add_module('1_2_conv_2', nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False))
		self.features.add_module('1_2_relu_2', nn.ReLU(inplace=True))
		self.features.add_module('1_2_maxpool', nn.MaxPool2d(kernel_size=2, stride=2))
		# conv_2_1
		self.features.add_module('2_1_conv_1', nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False))
		self.features.add_module('2_1_relu_1', nn.ReLU(inplace=True))
		self.features.add_module('2_1_conv_2', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
		self.features.add_module('2_1_relu_2', nn.ReLU(inplace=True))
		self.features.add_module('2_1_conv_3', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
		self.features.add_module('2_1_relu_3', nn.ReLU(inplace=True))
		self.features.add_module('2_1_maxpool', nn.MaxPool2d(kernel_size=2, stride=2))
		# conv_2_2
		self.features.add_module('2_2_conv_1', nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False))
		self.features.add_module('2_2_relu_1', nn.ReLU(inplace=True))
		self.features.add_module('2_2_conv_2', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False))
		self.features.add_module('2_2_relu_2', nn.ReLU(inplace=True))
		self.features.add_module('2_2_conv_3', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False))
		self.features.add_module('2_2_relu_3', nn.ReLU(inplace=True))
		self.features.add_module('2_2_maxpool', nn.MaxPool2d(kernel_size=2, stride=2))
		# conv_2_3
		self.features.add_module('2_3_conv_1', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False))
		self.features.add_module('2_3_relu_1', nn.ReLU(inplace=True))
		self.features.add_module('2_3_conv_2', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False))
		self.features.add_module('2_3_relu_2', nn.ReLU(inplace=True))
		self.features.add_module('2_3_conv_3', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False))
		self.features.add_module('2_3_relu_3', nn.ReLU(inplace=True))
		self.features.add_module('2_3_maxpool', nn.MaxPool2d(kernel_size=2, stride=2))
		self.avgpool = nn.AdaptiveAvgPool2d(output_size = (7, 7))
		self.classifier = nn.Sequential(
			nn.Linear(512 * 7 * 7, 4096),
			nn.ReLU(inplace=True),

			nn.Dropout(p = 0.5, inplace = False),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),

			nn.Dropout(p=0.5, inplace=False),
			nn.Linear(4096, num_classes),
		)
	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)
		x = x.view(x.size(0), 512 * 7 * 7) # resize for fc layers
		x = self.classifier(x)
		return x