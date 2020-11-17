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

import matplotlib.animation as animation
from matplotlib import style

from helper import *
from scipy.stats import norm #For our active learning algorithm

#We will use sklearn to create our Gaussian Process
import sklearn.gaussian_process as gp

#Make results reproducible
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)



string_states = np.load("string_states.npy")
labels = np.load("labels.npy")

#Shuffle the Data
np.random.seed(50) #Make this reproducible
shuffle_indices = np.random.permutation(len(string_states))
string_states = string_states[shuffle_indices]
labels = labels[shuffle_indices]

#Reduce the size of these datasets... we only need approximations for our function calls 2mil -> 100k
string_states = string_states[:100000]
labels = labels[:100000]

def create_fast(lr, lambd): #Single Layer Neural Network
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
	
	#model.summary() #Print a summary of the model
	return model

def res_block(X, lr, lambd, filters=256):
	#Convolutional Layers
	channel = Conv1D(filters=filters, kernel_size=3, strides=1, 
	kernel_initializer=glorot_uniform(), kernel_regularizer=l2(lambd), activation="relu", padding="causal")(X)

	channel = BatchNormalization(axis=-1)(channel)
	channel = Conv1D(filters=filters, kernel_size=3, strides=1, 
	kernel_initializer=glorot_uniform(), kernel_regularizer=l2(lambd), activation="linear", padding="causal")(channel)

	X = Add()([X, channel])
	X = BatchNormalization(axis=-1)(X)
	X = Activation("relu")(X)
	return X


def create_slow(lr, lambd, decay_rate, momentum, filters=128, block_num=10): #Convolutional Neural Network
	image = Input(shape=(10,26)) #String State Representation

	#Convolutional Layers
	X = Conv1D(filters=filters, kernel_size=3, strides=1, 
	kernel_initializer=glorot_uniform(), kernel_regularizer=l2(lambd), activation="relu", input_shape=(10,26), padding="causal")(image)

	for _ in range(block_num):
		X = res_block(X, lr, lambd, filters=filters)

	latent = Flatten()(X)
	latent = Dense(256, activation="relu", kernel_initializer=glorot_uniform(), 
		kernel_regularizer=l2(lambd))(latent)
	latent = Dense(128, activation="relu", kernel_initializer=glorot_uniform(), 
		kernel_regularizer=l2(lambd))(latent)

	p = Dense(26, activation="softmax", kernel_initializer=glorot_uniform(),
		kernel_regularizer=l2(lambd))(latent)

	#Create model now
	model = Model(inputs=image, outputs=p)

	#Compile model with our training hyperparameters
	optimizer = tf.keras.optimizers.SGD(lr, momentum=momentum, decay=decay_rate, nesterov=True)
	model.compile(loss="categorical_crossentropy", optimizer=optimizer, 
		metrics=[tf.keras.metrics.TopKCategoricalAccuracy(k=5)])
	
	#model.summary() #Print a summary of the model
	return model

def save_model(model, modelFile, weightFile): #Save model to json and weights to HDF5
		from tensorflow.keras.models import model_from_json
		model_json = model.to_json()
		with open(modelFile, "w") as json_file:
			json_file.write(model_json)
		model.save_weights(weightFile)
		print("Model saved!")

def train_model(lr, lambd, decay_rate, momentum, save=False, plot=False, load=True):
	model = create_slow(lr, lambd, decay_rate, momentum)
	if load:
		start_param = load_model("slowmodel.json", "slowmodel.h5") #Import the trained model
		model.set_weights(start_param.get_weights())

	loss_history = model.fit(string_states, labels, batch_size=1024, epochs=7, validation_split=0.1, verbose=False)

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


#Parameters of our Search
iterations = 100 #How many parameters will we search before making a conclusion

#Put nonsense values
hyper_params = []
topk = []
indices = [] #Save indices of used hyper params for plotting

current_lr = 3e-3
current_lambd = 1e-4
current_decay = current_lr / 7
current_momentum = 0.8

#Generate hyperparameter grid space we will optimize over.
max_lr = 1e-1
min_lr = 1e-6

max_lambd = 1e-1
min_lambd = 1e-5

max_decay = 1e-1
min_decay = 1e-6

max_momentum = 0.9
min_momentum = 0.5

resolution = 10 #create a resxres grid of values to evaluate

#Calculate exponential coefficients for spacing
lr_coeff = np.power(min_lr / max_lr, 1/resolution)
lambd_coeff = np.power(min_lambd / max_lambd, 1/resolution)
decay_coeff = np.power(min_decay/ max_decay, 1/resolution)

x_test = []
for x in range(resolution):
	lr = max_lr * np.power(lr_coeff, x)
	for y in range(resolution):
		lambd = max_lambd * np.power(lambd_coeff, y)
		for a in range(resolution):
			decay = max_decay * np.power(decay_coeff, a)
			for b in range(resolution):
				momentum = min_momentum + b * (max_momentum - min_momentum) / (resolution - 1)
				x_test.append([lr, lambd, decay, momentum])

x_test = np.reshape(x_test, [int(resolution*resolution*resolution*resolution), 4])

print("Testing: (", current_lr, ", ", current_lambd, ", ", current_decay, ", ", current_momentum, ")")

#Training Loop
plt.ion()
fig, axs = plt.subplots(2, figsize=(8,8))
for it in range(iterations):

	#Conduct k-fold CV to get an estimate of the generalization error of the model.

	_ , history = train_model(current_lr, current_lambd, current_decay, current_momentum, save=False, plot=False, load=False) #Train a model using hypothesized best hyper params

	hyper_params.append([current_lr, current_lambd, current_decay, current_momentum]) #Append hyper params we used to train
	topk.append(history.history["val_top_k_categorical_accuracy"][-1]) #Append final validation loss of model training
	print("--Actual Accuracy--: ", history.history["val_top_k_categorical_accuracy"][-1])

	if it > 1:
		#Create a gaussian process to model what hyperparams are the best
		#kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RBF(10.0, (1e-3, 1e3))
		kernel = gp.kernels.ConstantKernel(1.0) * gp.kernels.Matern(length_scale=1.0, nu=2.5)
		model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=True)

		x_train = np.reshape(hyper_params, [-1, 4])
		y_train = np.reshape(topk, [-1, 1])
		model.fit(x_train, y_train)
		mu, sigma = model.predict(x_test, return_std=True) #Make a prediction of the loss over param space

		"""
		lower_bound = np.squeeze(mu) - 0.6 * np.squeeze(sigma) #Here we minimize the lower confidence bound.
		index = np.argmin(lower_bound) #Find the parameters that minimize loss over the hyperparam space
		"""
		optimum = np.max(topk)

		sigma = np.squeeze(sigma)
		mu = np.squeeze(mu)

		unbound_Z = (mu - optimum)/sigma
		Z = np.where(sigma > 0, unbound_Z, 0) #Calculate Z for EI formula
		unbound_EI = (mu - optimum)*norm.cdf(Z) + sigma*norm.pdf(Z)
		EI = np.where(sigma > 0, unbound_EI, 0)
		index = np.argmax(EI)
		current_lr = x_test[index][0]
		current_lambd = x_test[index][1]
		current_decay = x_test[index][2]
		current_momentum = x_test[index][3]

		indices.append(index)

		#Plot our estimations and our acqusition function
		axs[0].clear()
		axs[1].clear()
		axs[0].plot(mu, 'b--', lw=2, label="CV Accuracy")
		axs[0].fill_between(np.arange(int(resolution*resolution*resolution*resolution))
			, mu + sigma, mu - sigma, alpha=0.2)
		axs[0].axhline(y=optimum, color="black", linestyle="--")
		axs[0].plot(indices, topk, 'kx')
		axs[0].legend()
		axs[1].plot(EI, 'r', label="Expected Improvement")
		axs[1].axvline(x=index, color="black", linestyle="--")
		axs[1].legend()
		plt.draw()
		plt.pause(0.0001)

		print("Testing: (", current_lr, ", ", current_lambd, ", ", current_decay, ", ", current_momentum, ")")
		print("Projected Accuracy: ", mu[index])
	else: #If we don't have enough samples for the gaussian process, randomly sample from the gridspace
		index = np.random.choice(int(resolution*resolution))
		current_lr = x_test[index][0]
		current_lambd = x_test[index][1]
		current_decay = x_test[index][2]
		current_momentum = x_test[index][3]
		indices.append(index)

		#Plot our estimations and our acqusition function
		axs[0].clear()
		axs[1].clear()
		axs[0].plot(indices, topk, 'kx', label="Samples")
		axs[0].legend()
		axs[1].axvline(x=index, color="black", linestyle="--", label="Next Sample")
		axs[1].legend()
		plt.draw()
		plt.pause(0.0001)

		print("Testing: (", current_lr, ", ", current_lambd, ", ", current_decay, ", ", current_momentum, ")")
		print("No Projected Accuracy (Random Sample)")

	


best_index = np.argmax(topk)
best_params = hyper_params[best_index]
print("Best Parameters: ", best_params)
print("Sample Performance: ", np.max(topk))

x_train = np.reshape(hyper_params, [-1, 4])
y_train = np.reshape(topk, [-1, 1])

np.save("hyp_params", x_train)
np.save("top-5-accuracy", y_train)





