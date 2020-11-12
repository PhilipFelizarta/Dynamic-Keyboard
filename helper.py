"""
This file will hold all the helper functions we need for encoding/decoding text into our 
numerical representations for the model we are using.
"""

def generate_dict(): #Generate dictionary that relates characters to their respective index
	enc_dict = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g":6, "h":7, "i":8, "j":9,
	"k":10, "l":11, "m":12, "n":13, "o":14, "p":15, "q":16, "r":17, "s":18, "t":19,
	"u":20, "v":21, "w":22, "x":23, "y":24, "z":25}

	return enc_dict


def encode_state(character_list, enc_dict):
	import numpy as np

	if len(character_list) > 10: #If our word is longer than 10 characters
		character_list = character_list[-10:] #Take the last ten characters

	string_state = np.zeros((10, 26))
	index = 0
	for char in character_list:
		onehot_ind = enc_dict[char] #We need to handle punctuation before...
		string_state[index][onehot_ind] = 1#Encode the string state matrix
		index += 1

	return string_state

def encode_label(character, enc_dict):
	import numpy as np
	
	label = np.zeros(26)
	ind = enc_dict[character] #Make sure character is not punctuation before calling method
	label[ind] = 1
	return label

def distr_to_radii(p, r_max=4, r_min=1):
	radii = p * (r_max - r_min) + r_min #Numpy will broadcast the addition and multiplcation
	return radii 

def save_model(model, modelFile, weightFile): #Save model to json and weights to HDF5
	from tensorflow.keras.models import model_from_json
	model_json = model.to_json()
	with open(modelFile, "w") as json_file:
		json_file.write(model_json)
	model.save_weights(weightFile)
	print("Model saved!")

def load_model(modelFile, weightFile, update=False): #load model from json and HDF5
	from tensorflow.keras.models import model_from_json


	json_file = open(modelFile, 'r')
	load_model_json = json_file.read()
	json_file.close()
	if not update:
		load_model = model_from_json(load_model_json)
		
	load_model.load_weights(weightFile)
	print("Model Loaded!")
	return load_model