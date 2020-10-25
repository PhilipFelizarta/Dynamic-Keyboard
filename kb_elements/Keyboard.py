from p5 import *
from Button import *

button = Button((150, 300), 60, "a")
button2 = Button((50, 250), 30, "b")

def setup():
	size(250, 400) #dimensions of a Galaxy S9
	button.setupButton()

def draw():
	
	background(255) #white background for now

	button.drawButton()
	button2.drawButton()

if __name__ == '__main__':
	run()