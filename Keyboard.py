from kb_elements.Button import *
from kb_elements.Display import *
from predictive_model.helper import *
import tensorflow as tf
import numpy as np

text_string = []
button_list = [None] * 29
radii = np.full(26, 26)
model = load_model("predictive_model/slowmodel.json", "predictive_model/slowmodel.h5") #Import the trained model
alphabet_dict = generate_dict()

string_state = encode_state(text_string, alphabet_dict) #Convert typed text into a matrix for model input
string_state = np.reshape(string_state, [1, 10, 26])

distr = model.predict(string_state) #Call a prediction at initialization to get the machine runnin

def populateAlphabet(button, radii):
	#All 26 letter buttons.
	button[0] = Button((95, 250), radii[0], "a")
	button[1] = Button((95, 175), radii[1], "b")
	button[2] = Button((125, 195), radii[2], "c")
	button[3] = Button((220, 250), radii[3], "d")
	button[4] = Button((155, 250), radii[4], "e")
	button[5] = Button((63, 230), radii[5], "f")
	button[6] = Button((95, 213), radii[6], "g")
	button[7] = Button((30, 250), radii[7], "h")
	button[8] = Button((185, 265), radii[8], "i")
	button[9] = Button((95, 320), radii[9], "j")
	button[10] = Button((125, 305), radii[10], "k")
	button[11] = Button((185, 230), radii[11], "l")
	button[12] = Button((155, 285), radii[12], "m") 
	button[13] = Button((95, 285), radii[13], "n")
	button[14] = Button((125, 230), radii[14], "o")
	button[15] = Button((125, 155), radii[15], "p")
	button[16] = Button((125, 340), radii[16], "q")
	button[17] = Button((155, 213), radii[17], "r")
	button[18] = Button((125, 270), radii[18], "s")
	button[19] = Button((63, 305), radii[19], "t")
	button[20] = Button((63, 270), radii[20], "u")
	button[21] = Button((155, 320), radii[21], "v")
	button[22] = Button((185, 305), radii[22], "w")
	button[23] = Button((63, 195), radii[23], "x")
	button[24] = Button((155, 175), radii[24], "y")
	button[25] = Button((185, 195), radii[25], "z")

	button[26] = Button((220, 130), 45, "<-")
	button[27] = Button((220, 365), 45, "<-|")
	button[28] = Button((30, 365), 45, "\' \'")

def setup():
	size(250, 400) #dimensions of a Galaxy S9
	
def draw():
	background(255) #white background for now
	string_by = ""

	#Letter body
	stroke(0)
	circle((125, 250), 230)

	#Display box and text
	display = Display((10, 10)); display.drawTextBox((230, 50)); display.drawText(string_by.join(text_string), (15, 45))
	populateAlphabet(button_list, radii)
	
	for x in range(0, len(button_list)):
		button_list[x].drawButton()

def mouse_released():
	global text_string
	global radii 

	for i in range(0, len(button_list)):
		button_list[i].pressKey(text_string)


	#Preprocessing to take the last word in the sentence
	last_word = []
	for char in text_string:
		last_word.append(char)
		if char == " ":
			last_word = []

	string_state = encode_state(last_word, alphabet_dict) #Convert typed text into a matrix for model input
	string_state = np.reshape(string_state, [1, 10, 26])

	distr = model.predict(string_state) #Get our prediction of likelihoods
	distr = np.squeeze(distr) #Get rid of extra dimension [1,26] -> [26]
	radii = distr_to_radii(distr, r_max=45, r_min=25)

if __name__ == '__main__':
	run()