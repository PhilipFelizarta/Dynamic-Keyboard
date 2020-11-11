import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Remove logging messages in terminal

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, Dropout, AlphaDropout, Lambda, Attention
from tensorflow.keras.layers import GlobalAveragePooling2D, Multiply, Permute, Reshape
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

#We will use sklearn to create our Gaussian Process
import sklearn.gaussian_process as gp

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
	
	#model.summary() #Print a summary of the model
	return model

def save_model(model, modelFile, weightFile): #Save model to json and weights to HDF5
		from tensorflow.keras.models import model_from_json
		model_json = model.to_json()
		with open(modelFile, "w") as json_file:
			json_file.write(model_json)
		model.save_weights(weightFile)
		print("Model saved!")

def train_model(lr, lambd, save=False, plot=False):
	model = create_fast(lr, lambd)
	loss_history = model.fit(string_states, labels, batch_size=64, epochs=20, validation_split=0.05, verbose=False)

	if save:
		save_model(model, "fastModel.json", "fastModel.h5") #Save model

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


iterations = 100
hyper_params = [[3e-3, 1e-4]]
val_loss = [1.8239]

current_lr = 3e-4
current_lambd = 1e-4

#Generate hyperparameter grid space we will optimize over.
max_lr = 1e-1
max_lambd = 1e-1
resolution = 100 #create a resxres grid of values to evaluate
x_test = []
for x in range(resolution):
	lr = max_lr / np.power(1.2, x)
	for y in range(resolution):
		lambd = max_lambd / np.power(1.2, y)
		x_test.append([lr, lambd])

x_test = np.reshape(x_test, [int(resolution*resolution), 2])

print("Testing: (", current_lr, ", ", current_lambd, ")")
for it in range(iterations):
	_ , history = train_model(current_lr, current_lambd, save=False, plot=False) #Train a model using hypothesized best hyper params

	hyper_params.append([current_lr, current_lambd]) #Append hyper params we used to train
	val_loss.append(history.history["val_loss"][-1]) #Append final validation loss of model training
	print("Actual Loss: ", history.history["val_loss"][-1])

	#Create a gaussian process to model what hyperparams are the best
	kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RBF(10.0, (1e-3, 1e3))
	model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=True)

	x_train = np.reshape(hyper_params, [-1, 2])
	y_train = np.reshape(val_loss, [-1, 1])
	model.fit(x_train, y_train)
	y_pred, std = model.predict(x_test, return_std=True) #Make a prediction of the loss over param space


	lower_bound = np.squeeze(y_pred) - 2 * np.squeeze(std)
	index = np.argmin(lower_bound) #Find the parameters that minimize loss over the hyperparam space
	current_lr = x_test[index][0]
	current_lambd = x_test[index][1]
	print("Testing: (", current_lr, ", ", current_lambd, ")")
	print("Projected Loss: ", y_pred[index])


best_index = np.argmin(val_loss)
best_params = hyper_params[best_index]
print("Best Parameters: ", best_params)

x_train = np.reshape(hyper_params, [-1, 2])
y_train = np.reshape(val_loss, [-1, 1])

np.save("hyp_params", x_train)
np.save("real_loss", y_train)





