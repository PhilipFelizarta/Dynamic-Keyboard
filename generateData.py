import numpy as np 
import helper
from time import time

#This file should convert the text file into the actual numpy arrrays we will use for training the NN
f = open("words_alpha.txt", "r", encoding="ascii") #Open our text file


enc_dict = helper.generate_dict()
t0 = time() #time how long it takes to generate the dataset

string_states = [] #List of string_states for input
labels = [] #List of labels... note both lists must have same len()

for line in f: #Iterate through each line of the file
	char_list = list(line)
	length = len(char_list) - 1
	char_list.pop() #Remove the last \n

	if length > 11: #We take 11 characters since input is 10 and label is 1
		char_list = char_list[-11:]

	if length > 1: #Don't take single characters for potential words
		lab_char = char_list.pop() #Last character in substring is the label
		substring = char_list
		state = helper.encode_state(substring, enc_dict)
		label = helper.encode_label(lab_char, enc_dict)
		string_states.append(state)
		labels.append(label)
f.close()

string_states = np.reshape(np.array(string_states), [-1, 10, 26])
labels = np.reshape(np.array(labels), [-1, 26])

print(string_states.shape)
print(labels.shape)

np.save("string_states", string_states)
np.save("labels", labels)
t1 = time()
print("Time: ", t1 - t0)
