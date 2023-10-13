import argparse
import tensorflow as tf
import tensorflow_addons as tfa
import torch

def conv_tf(kernels,kernelsize,stride,pad,name):
	layer = []
	layer.append(tf.keras.layers.Conv2D(kernels,kernelsize,strides=stride,padding=pad))
	layer.append(tf.keras.layers.BatchNormalization())
	layer.append(tf.keras.layers.ReLU())
	return tf.keras.Sequential(layer,name='conv_bn_relu_'+str(name))

def conv_mp_tf(kernels,kernelsize,stride,pad,poolsize,poolstride,name):
	layer = []
	layer.append(tf.keras.layers.Conv2D(kernels,kernelsize,strides=stride,padding=pad))
	layer.append(tf.keras.layers.BatchNormalization())
	layer.append(tf.keras.layers.ReLU())
	layer.append(tf.keras.layers.MaxPooling2D(pool_size=(poolsize,poolsize),strides=poolstride))
	return tf.keras.Sequential(layer,name='conv_bn_relu_mp_'+str(name))

class VGG_tf(tf.keras.Model):
	def __init__(self):
		super().__init__()
		self.conv1 = conv_mp_tf(64,3,1,'SAME',2,2,1)
		self.conv2 = conv_mp_tf(128,3,1,'SAME',2,2,2)
		self.conv3 = conv_tf(256,3,1,'SAME',3)
		self.conv4 = conv_mp_tf(256,3,1,'SAME',2,2,4)
		self.conv5 = conv_tf(512,3,1,'SAME',5)
		self.conv6 = conv_mp_tf(512,3,1,'SAME',2,2,6)
		self.conv7 = conv_tf(512,3,1,'SAME',7)
		self.conv8 = conv_mp_tf(512,3,1,'SAME',2,2,8)
		self.avgpool = tfa.layers.AdaptiveAveragePooling2D(output_size=7)
		self.dense1 = tf.keras.layers.Dense(4096)
		self.relu9 = tf.keras.layers.ReLU()
		self.drop1 = tf.keras.layers.Dropout(rate=0.5)
		self.dense2 = tf.keras.layers.Dense(4096)
		self.relu10 = tf.keras.layers.ReLU()
		self.drop2 = tf.keras.layers.Dropout(rate=0.5)
		self.classifier = tf.keras.layers.Dense(1000,activation='sigmoid')

	def call(self,x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.conv5(x)
		x = self.conv6(x)
		x = self.conv7(x)
		x = self.conv8(x)
		x = self.avgpool(x)
		x = self.drop1(self.relu9(self.dense1(x)))
		x = self.drop2(self.relu10(self.dense2(x)))
		x = self.classifier(x)
		return x

class VGG_torch(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = torch.nn.Conv2d(3,64,3,stride=1,padding=1)
		self.bn1 = torch.nn.BatchNorm2d(64)
		self.relu1 = torch.nn.ReLU()
		self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2,stride=2)
		self.conv2 = torch.nn.Conv2d(64,128,3,stride=1,padding=1)
		self.bn2 = torch.nn.BatchNorm2d(128)
		self.relu2 = torch.nn.ReLU()
		self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2,stride=2)
		self.conv3 = torch.nn.Conv2d(128,256,3,stride=1,padding=1)
		self.bn3 = torch.nn.BatchNorm2d(256)
		self.relu3 = torch.nn.ReLU()
		self.conv4 = torch.nn.Conv2d(256,256,3,stride=1,padding=1)
		self.bn4 = torch.nn.BatchNorm2d(256)
		self.relu4 = torch.nn.ReLU()
		self.maxpool3 = torch.nn.MaxPool2d(kernel_size=2,stride=2)
		self.conv5 = torch.nn.Conv2d(256,512,3,stride=1,padding=1)
		self.bn5 = torch.nn.BatchNorm2d(512)
		self.relu5 = torch.nn.ReLU()
		self.conv6 = torch.nn.Conv2d(512,512,3,stride=1,padding=1)
		self.bn6 = torch.nn.BatchNorm2d(512)
		self.relu6 = torch.nn.ReLU()
		self.maxpool4 = torch.nn.MaxPool2d(kernel_size=2,stride=2)
		self.conv7 = torch.nn.Conv2d(512,512,3,stride=1,padding=1)
		self.bn7 = torch.nn.BatchNorm2d(512)
		self.relu7 = torch.nn.ReLU()
		self.conv8 = torch.nn.Conv2d(512,512,3,stride=1,padding=1)
		self.bn8 = torch.nn.BatchNorm2d(512)
		self.relu8 = torch.nn.ReLU()
		self.maxpool5 = torch.nn.MaxPool2d(kernel_size=2,stride=2)
		self.avgpool = torch.nn.AdaptiveAvgPool2d((7,7))
		self.dense1 = torch.nn.Linear(512*7*7,4096)
		self.relu9 = torch.nn.ReLU()
		self.drop1 = torch.nn.Dropout()
		self.dense2 = torch.nn.Linear(4096,4096)
		self.relu10 = torch.nn.ReLU()
		self.drop2 = torch.nn.Dropout()
		self.classifier = torch.nn.Linear(4096,1000)
		
	def forward(self, x):
		x = self.maxpool1(self.relu1(self.bn1(self.conv1(x))))
		x = self.maxpool2(self.relu2(self.bn2(self.conv2(x))))
		x = self.relu3(self.bn3(self.conv3(x)))
		x = self.maxpool3(self.relu4(self.bn4(self.conv4(x))))
		x = self.relu5(self.bn5(self.conv5(x)))
		x = self.maxpool4(self.relu6(self.bn6(self.conv6(x))))
		x = self.relu7(self.bn7(self.conv7(x)))
		x = self.maxpool5(self.relu8(self.bn8(self.conv8(x))))
		x = self.avgpool(x)
		x = self.drop1(self.relu9(self.dense1(x)))
		x = self.drop2(self.relu10(self.dense2(x)))
		x = self.classifier(x)
		return x

def model_torch():
	return VGG_torch()

def model_tf():
	return VGG_tf()

def main(args):
	if args.model=='tf':
		print('Model will be created in Tensorflow')
		if args.depth=='11':
			model = model_tf()
			model.build(input_shape=(None,224,224,3))
			model.summary()
	else:
		print('Model will be created in Pytorch')
		if args.depth=='11':
			model = model_torch()
			print(model)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Create VGG model in Tensorflow or Pytorch')
	parser.add_argument('--model',
	                    default='tf',
	                    choices=['tf', 'torch'],
	                    help='Model will be created on Tensorflow, Pytorch (default: %(default)s)')
	parser.add_argument('--depth',
	                    default='11',
	                    choices=['11', '13', '16', '19'],
	                    help='VGG model depth (default: %(default)s)')
	args = parser.parse_args()
	main(args)