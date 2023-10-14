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

def conv_torc(in_c,out_c,kernelsize,stride,pad): #kernels,kernelsize,stride,pad,name):
	layer = torch.nn.Sequential(
			torch.nn.Conv2d(in_c,out_c,kernelsize,stride=stride,padding=pad),
			torch.nn.BatchNorm2d(out_c),
			torch.nn.ReLU()
		)
	return layer

def conv_mp_torc(in_c,out_c,kernelsize,stride,pad,mpkernels,mpstride): #kernels,kernelsize,stride,pad,name):
	layer = torch.nn.Sequential(
			torch.nn.Conv2d(in_c,out_c,kernelsize,stride=stride,padding=pad),
			torch.nn.BatchNorm2d(out_c),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(kernel_size=mpkernels,stride=mpstride)
		)
	return layer

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

class VGG_tf13(tf.keras.Model):
	def __init__(self):
		super().__init__()
		self.conv1 = conv_tf(64,3,1,'SAME',1)
		self.conv2 = conv_mp_tf(64,3,1,'SAME',2,2,2)
		self.conv3 = conv_tf(128,3,1,'SAME',3)
		self.conv4 = conv_mp_tf(128,3,1,'SAME',2,2,4)
		self.conv5 = conv_tf(256,3,1,'SAME',5)
		self.conv6 = conv_mp_tf(256,3,1,'SAME',2,2,6)
		self.conv7 = conv_tf(512,3,1,'SAME',7)
		self.conv8 = conv_mp_tf(512,3,1,'SAME',2,2,8)
		self.conv9 = conv_tf(512,3,1,'SAME',9)
		self.conv10 = conv_mp_tf(512,3,1,'SAME',2,2,10)
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
		x = self.conv9(x)
		x = self.conv10(x)
		x = self.avgpool(x)
		x = self.drop1(self.relu9(self.dense1(x)))
		x = self.drop2(self.relu10(self.dense2(x)))
		x = self.classifier(x)
		return x

class VGG_tf16(tf.keras.Model):
	def __init__(self):
		super().__init__()
		self.conv1 = conv_tf(64,3,1,'SAME',1)
		self.conv2 = conv_mp_tf(64,3,1,'SAME',2,2,2)
		self.conv3 = conv_tf(128,3,1,'SAME',3)
		self.conv4 = conv_mp_tf(128,3,1,'SAME',2,2,4)
		self.conv5 = conv_tf(256,3,1,'SAME',5)
		self.conv6 = conv_tf(256,3,1,'SAME',6)
		self.conv7 = conv_mp_tf(256,3,1,'SAME',2,2,7)
		self.conv8 = conv_tf(512,3,1,'SAME',8)
		self.conv9 = conv_tf(512,3,1,'SAME',9)
		self.conv10 = conv_mp_tf(512,3,1,'SAME',2,2,10)
		self.conv11 = conv_tf(512,3,1,'SAME',11)
		self.conv12 = conv_tf(512,3,1,'SAME',12)
		self.conv13 = conv_mp_tf(512,3,1,'SAME',2,2,13)
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
		x = self.conv9(x)
		x = self.conv10(x)
		x = self.conv11(x)
		x = self.conv12(x)
		x = self.conv13(x)
		x = self.avgpool(x)
		x = self.drop1(self.relu9(self.dense1(x)))
		x = self.drop2(self.relu10(self.dense2(x)))
		x = self.classifier(x)
		return x

class VGG_tf19(tf.keras.Model):
	def __init__(self):
		super().__init__()
		self.conv1 = conv_tf(64,3,1,'SAME',1)
		self.conv2 = conv_mp_tf(64,3,1,'SAME',2,2,2)
		self.conv3 = conv_tf(128,3,1,'SAME',3)
		self.conv4 = conv_mp_tf(128,3,1,'SAME',2,2,4)
		self.conv5 = conv_tf(256,3,1,'SAME',5)
		self.conv6 = conv_tf(256,3,1,'SAME',6)
		self.conv6 = conv_tf(256,3,1,'SAME',7)
		self.conv7 = conv_mp_tf(256,3,1,'SAME',2,2,8)
		self.conv8 = conv_tf(512,3,1,'SAME',9)
		self.conv9 = conv_tf(512,3,1,'SAME',10)
		self.conv9 = conv_tf(512,3,1,'SAME',11)
		self.conv10 = conv_mp_tf(512,3,1,'SAME',2,2,12)
		self.conv11 = conv_tf(512,3,1,'SAME',13)
		self.conv12 = conv_tf(512,3,1,'SAME',14)
		self.conv12 = conv_tf(512,3,1,'SAME',15)
		self.conv13 = conv_mp_tf(512,3,1,'SAME',2,2,16)
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
		x = self.conv9(x)
		x = self.conv10(x)
		x = self.conv11(x)
		x = self.conv12(x)
		x = self.conv13(x)
		x = self.avgpool(x)
		x = self.drop1(self.relu9(self.dense1(x)))
		x = self.drop2(self.relu10(self.dense2(x)))
		x = self.classifier(x)
		return x

class VGG_torch(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = conv_mp_torc(3,64,3,1,1,2,2)
		self.conv2 = conv_mp_torc(64,128,3,1,1,2,2)
		self.conv3 = conv_torc(128,256,3,1,1)
		self.conv4 = conv_mp_torc(256,256,3,1,1,2,2)
		self.conv5 = conv_torc(256,512,3,1,1)
		self.conv6 = conv_mp_torc(512,512,3,1,1,2,2)
		self.conv7 = conv_torc(512,512,3,1,1)
		self.conv8 = conv_mp_torc(512,512,3,1,1,2,2)
		self.avgpool = torch.nn.AdaptiveAvgPool2d((7,7))
		self.dense1 = torch.nn.Linear(512*7*7,4096)
		self.relu9 = torch.nn.ReLU()
		self.drop1 = torch.nn.Dropout()
		self.dense2 = torch.nn.Linear(4096,4096)
		self.relu10 = torch.nn.ReLU()
		self.drop2 = torch.nn.Dropout()
		self.classifier = torch.nn.Linear(4096,1000)
		
	def forward(self, x):
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

class VGG_torch13(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = conv_torc(3,64,3,1,1)
		self.conv2 = conv_mp_torc(64,64,3,1,1,2,2)
		self.conv3 = conv_torc(64,128,3,1,1)
		self.conv4 = conv_mp_torc(128,128,3,1,1,2,2)
		self.conv5 = conv_torc(128,256,3,1,1)
		self.conv6 = conv_mp_torc(256,256,3,1,1,2,2)
		self.conv7 = conv_torc(256,512,3,1,1)
		self.conv8 = conv_mp_torc(512,512,3,1,1,2,2)
		self.conv9 = conv_torc(512,512,3,1,1)
		self.conv10 = conv_mp_torc(512,512,3,1,1,2,2)
		self.avgpool = torch.nn.AdaptiveAvgPool2d((7,7))
		self.dense1 = torch.nn.Linear(512*7*7,4096)
		self.relu9 = torch.nn.ReLU()
		self.drop1 = torch.nn.Dropout()
		self.dense2 = torch.nn.Linear(4096,4096)
		self.relu10 = torch.nn.ReLU()
		self.drop2 = torch.nn.Dropout()
		self.classifier = torch.nn.Linear(4096,1000)
		
	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.conv5(x)
		x = self.conv6(x)
		x = self.conv7(x)
		x = self.conv8(x)
		x = self.conv9(x)
		x = self.conv10(x)
		x = self.avgpool(x)
		x = self.drop1(self.relu9(self.dense1(x)))
		x = self.drop2(self.relu10(self.dense2(x)))
		x = self.classifier(x)
		return x

def model_torch():
	return VGG_torch()

def model_torch13():
	return VGG_torch13()

def model_tf_11():
	return VGG_tf()

def model_tf_13():
	return VGG_tf13()

def model_tf_16():
	return VGG_tf16()

def model_tf_19():
	return VGG_tf19()

def main(args):
	if args.model=='tf':
		print('Model VGG_'+str(args.depth)+' will be created in Tensorflow')
		if args.depth=='11':
			model = model_tf_11()
			model.build(input_shape=(None,224,224,3))
			model.summary()
		elif args.depth=='13':
			model = model_tf_13()
			model.build(input_shape=(None,224,224,3))
			model.summary()
		elif args.depth=='16':
			model = model_tf_16()
			model.build(input_shape=(None,224,224,3))
			model.summary()
		elif args.depth=='19':
			model = model_tf_19()
			model.build(input_shape=(None,224,224,3))
			model.summary()
	else:
		print('Model VGG_'+str(args.depth)+' will be created in Pytorch')
		if args.depth=='11':
			model = model_torch()
			print(model)
		elif args.depth=='13':
			model = model_torch13()
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