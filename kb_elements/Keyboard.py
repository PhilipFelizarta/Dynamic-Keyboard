from Button import *
from Display import *

text_string = []

def setup():
	size(250, 400) #dimensions of a Galaxy S9
	

def draw():
	background(255) #white background for now
	string_join = ""

	stroke(0)
	circle((125, 250), 230)

	display = Display((10, 10)); display.drawTextBox((230, 50)); display.drawText(string_join.join(text_string), (15, 45))

	buttonA = Button((150, 300), 60, "a"); buttonA.drawButton()
	buttonB = Button((50, 250), 30, "b"); buttonB.drawButton()
	buttonC = Button((170, 200), 45, "c"); buttonC.drawButton()

def mouse_released():
	global text_string
	text_string.append("a")

if __name__ == '__main__':
	run()