from Button import *
from Display import *

text_string = []
button_list = [None] * 26

def populateAlphabet(button, diameter):
	#All 26 letter buttons.
	button[0] = Button((95, 250), diameter, "a")
	button[1] = Button((95, 175), diameter, "b")
	button[2] = Button((125, 195), diameter, "c")
	button[3] = Button((220, 250), diameter, "d")
	button[4] = Button((155, 250), diameter, "e")
	button[5] = Button((63, 230), diameter, "f")
	button[6] = Button((95, 213), diameter, "g")
	button[7] = Button((30, 250), diameter, "h")
	button[8] = Button((185, 305), diameter, "i")
	button[9] = Button((95, 320), diameter, "j")
	button[10] = Button((125, 305), diameter, "k")
	button[11] = Button((185, 230), diameter, "l")
	button[12] = Button((155, 285), diameter, "m") 
	button[13] = Button((95, 285), diameter, "n")
	button[14] = Button((125, 230), diameter, "o")
	button[15] = Button((125, 155), diameter, "p")
	button[16] = Button((125, 340), diameter, "q")
	button[17] = Button((155, 213), diameter, "r")
	button[18] = Button((125, 270), diameter, "s")
	button[19] = Button((63, 305), diameter, "t")
	button[20] = Button((63, 270), diameter, "u")
	button[21] = Button((155, 320), diameter, "v")
	button[22] = Button((185, 265), diameter, "w")
	button[23] = Button((63, 195), diameter, "x")
	button[24] = Button((155, 175), diameter, "y")
	button[25] = Button((185, 195), diameter, "z")

def setup():
	size(250, 400) #dimensions of a Galaxy S9
	
def draw():
	background(255) #white background for now
	string_join = ""

	#Letter body
	stroke(0)
	circle((125, 250), 230)

	#Display box and text
	display = Display((10, 10)); display.drawTextBox((230, 50)); display.drawText(string_join.join(text_string), (15, 45))

	populateAlphabet(button_list, 30)
	
	for x in range(0, len(button_list)):
		button_list[x].drawButton()

	#TODO: Special button implementation
	caps_button = Button((30, 130), 45, "^"); caps_button.drawButton()
	backspace_button = Button((220, 130), 45, "<-"); backspace_button.drawButton()
	return_button = Button((220, 365), 45, "<-|"); return_button.drawButton()
	space_button = Button((30, 365), 45, "\' \'"); space_button.drawButton()

def mouse_released():
	global text_string
	for i in range(0, len(button_list)):
		button_list[i].pressKey(text_string)

if __name__ == '__main__':
	run()