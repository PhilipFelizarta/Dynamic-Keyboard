from Button import *

def setup():
	size(250, 400) #dimensions of a Galaxy S9

def draw():
	
	background(255) #white background for now

	button = Button((150, 300), 60, "a"); button.drawButton()
	button2 = Button((50, 250), 30, "b"); button2.drawButton()
	button3 = Button((170, 150), 45, "c"); button3.drawButton()


if __name__ == '__main__':
	run()