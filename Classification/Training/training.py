import matplotlib
matplotlib.use("Agg")

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from Classification.Training.architecture.LeNet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
from osgeo import gdal

class training(object):
	def __init__(self, path_to_dataset, save_to_dir_model, lbl, bs, epochs, lr, seed, numclasses, sizeofimage, splitDataset):
		self.path_to_dataset = path_to_dataset
		self.save_to_dir_model = save_to_dir_model
		self.lbl = lbl
		self.bs = bs
		self.epochs = epochs
		self.lr = lr
		self.seed = seed
		self.numclasses = numclasses
		self.sizeofimage = sizeofimage
		self.test_size = splitDataset

	def labeled(self,label):
		return self.lbl.get(label)

	def train(self):
		path_code = os.getcwd()
		BS = self.bs
		EPOCHS = self.epochs
		INIT_LR = self.lr
		seed = self.seed
		split_test_size = self.test_size
		total_classes = self.numclasses
		data = []
		labels = []
		CanalesImagen=None
		imagePaths = sorted(list(paths.list_images(self.path_to_dataset)))
		random.seed(seed)
		random.shuffle(imagePaths)

		for imagePath in imagePaths:
			filename = os.path.join(path_code, imagePath)
			
			raster = gdal.Open(filename)
			if raster.RasterCount == 1:
				image = raster.ReadAsArray()
			else:
				bandasIm = []
				for i in range(raster.RasterCount):
					bandasIm.append(raster.GetRasterBand(i+1).ReadAsArray())
				image = cv2.merge(bandasIm)

			#image = cv2.imread(filename,-1)
			CanalesImagen=raster.RasterCount
			image = img_to_array(image)
			data.append(image)

			label = imagePath.split(os.path.sep)[-2]
			label = self.labeled(label)
			labels.append(label)

		data = np.array(data, dtype="float")
		labels = np.array(labels)

		(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=split_test_size, random_state=seed)

		trainY = to_categorical(trainY, num_classes=total_classes)
		testY = to_categorical(testY, num_classes=total_classes)

		aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")
		model = LeNet.build(width=self.sizeofimage, height=self.sizeofimage, depth=CanalesImagen, classes=total_classes)
		opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
		#opt = Adamax(learning_rate=INIT_LR, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Adamax")
		model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]) 
		
		#w data_aug
		H = model.fit(x=aug.flow(trainX, trainY, batch_size=BS), validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,	epochs=EPOCHS, verbose=1)
		#w/out data_aug
		##H = model.fit(x=trainX, y = trainY, batch_size=BS, validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS, epochs=EPOCHS, verbose=1)

		model.save(self.save_to_dir_model, include_optimizer=False)
		plt.style.use("ggplot")
		plt.figure()
		N = EPOCHS
		plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
		plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
		plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
		plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
		plt.title("Training Loss and Accuracy on dataset General")
		plt.xlabel("Epoch #")
		plt.ylabel("Loss/Accuracy")
		plt.legend(loc="lower left")
		plt.savefig("training_results_general.png")