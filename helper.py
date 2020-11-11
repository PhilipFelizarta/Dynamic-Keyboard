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