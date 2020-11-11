from Button import *
from Display import *

def setup():
	size(250, 400) #dimensions of a Galaxy S9

def draw():
	
	background(255) #white background for now
	text_string = ""

	display = Display(20, 50, "Hello"); display.drawTextBox((10, 10), (230, 50)); display.drawText(text_string, (15, 45))

	button = Button((150, 300), 60, "a"); button.drawButton()
	button2 = Button((50, 250), 30, "b"); button2.drawButton()
	button3 = Button((170, 150), 45, "c"); button3.drawButton()

	button.type(text_string)


if __name__ == '__main__':
	run()