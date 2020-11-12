import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Remove logging messages in terminal

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, Dropout, AlphaDropout, Lambda, Attention
from tensorflow.keras.layers import GlobalAveragePooling2D, Multiply, Permute, Reshape, Conv1D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

#Make results reproducible
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)



string_states = np.load("string_states.npy")
labels = np.load("labels.npy")

def create_fast(lr, lambd):
	image = Input(shape=(10,26)) #String State Representation
	X = Flatten()(image) #NN will take advantage of positional data... try this in CNN
	latent = Dense(1024, activation="relu", kernel_initializer=glorot_uniform(), 
		kernel_regularizer=l2(lambd))(X)
	p = Dense(26, activation="softmax", kernel_initializer=glorot_uniform(),
		kernel_regularizer=l2(lambd))(latent)

	#Create model now
	model = Model(inputs=image, outputs=p)

	#Compile model with our training hyperparameters
	optimizer = tf.keras.optimizers.SGD(lr, momentum=0.9)
	model.compile(loss="categorical_crossentropy", optimizer=optimizer, 
		metrics=[tf.keras.metrics.TopKCategoricalAccuracy(k=5)])
	
	model.summary() #Print a summary of the model
	return model

def create_slow(lr, lambd, filters=128): #Convolutional Neural Network
	image = Input(shape=(10,26)) #String State Representation

	#Convolutional Layers
	X = Conv1D(filters=filters, kernel_size=3, strides=1, 
	kernel_initializer=glorot_uniform(), kernel_regularizer=l2(lambd), activation="relu")(image)

	X = Conv1D(filters=filters, kernel_size=3, strides=1, 
	kernel_initializer=glorot_uniform(), kernel_regularizer=l2(lambd), activation="relu")(X)

	latent = Flatten()(X)
	latent = Dense(256, activation="relu", kernel_initializer=glorot_uniform(), 
		kernel_regularizer=l2(lambd))(latent)
	p = Dense(26, activation="softmax", kernel_initializer=glorot_uniform(),
		kernel_regularizer=l2(lambd))(latent)

	#Create model now
	model = Model(inputs=image, outputs=p)

	#Compile model with our training hyperparameters
	optimizer = tf.keras.optimizers.SGD(lr, momentum=0.9)
	model.compile(loss="categorical_crossentropy", optimizer=optimizer, 
		metrics=[tf.keras.metrics.TopKCategoricalAccuracy(k=5)])
	
	model.summary() #Print a summary of the model
	return model

def save_model(model, modelFile, weightFile): #Save model to json and weights to HDF5
		from tensorflow.keras.models import model_from_json
		model_json = model.to_json()
		with open(modelFile, "w") as json_file:
			json_file.write(model_json)
		model.save_weights(weightFile)
		print("Model saved!")

def train_model(lr, lambd, save=False, plot=False):
	model = create_slow(lr, lambd)
	loss_history = model.fit(string_states, labels, batch_size=64, epochs=50, validation_split=0.05, verbose=True)

	if save:
		save_model(model, "slowModel.json", "slowModel.h5") #Save model

	if plot:
		plt.plot(loss_history.history['loss'], label='Crossentropy Loss (training data)')
		plt.plot(loss_history.history['val_loss'], label='Crossentropy Loss (validation data)')
		plt.title('Crossentropy Loss for 1 hidden layer NN')
		plt.ylabel('CE Loss')
		plt.xlabel('Epochs')
		plt.legend(loc="upper left")
		plt.show()


		plt.plot(loss_history.history['top_k_categorical_accuracy'], label='Top-5 Accuracy (training data)')
		plt.plot(loss_history.history['val_top_k_categorical_accuracy'], label='Top-5 Accuracy (validation data)')
		plt.title('Top-5 Accuracy for 1 hidden layer NN')
		plt.ylabel('Top-5 Accuracy')
		plt.xlabel('Epochs')
		plt.legend(loc="upper left")
		plt.show()


	return model, loss_history

#Use hyper parameters found in parameterSearch
lr = 0.004507324
lambd = 1.44896e-09

model, loss_history = train_model(lr, lambd, save=True, plot=True)