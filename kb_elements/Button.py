from p5 import *

'''
class Button:
	def __init__(self, coordinates, diameter, character, font_name:
		self.coordinates = coordinates
		self.diameter = diameter
		self.character = character
		self.font_name = font_name
		font = create_font(self.font_name, 10)
		text_font(font)
		text_align(CENTER, CENTER)

	def drawKey(self, fontName):
		text_font(self.fontName)
		text_align(CENTER, CENTER)

		if mouse_is_pressed:
			fill(125)
		else:
			fill(155)
		circle(self.__coordinates, self.__diameter)

		fill(255)
		text(self.__character, self.__coordinates)
'''

def setup():
	size(250, 400)
	arial = create_font("arial.ttf", 10)
	text_font(arial)
	text_align(CENTER, CENTER)

def debug(coordinates, dist_from_key):
	fill(0)
	text_size(15)
	text("Position of mouseY\nrelative to canvas is: " + str(mouse_y), (120, 40)); 

	text_size(15)
	text("Position of mouseX\nrelative to canvas is: " + str(mouse_x), (120, 80)); 

	text_size(15)
	text("Distance from key is: " + str(dist_from_key), (120, 120))

def draw(circle_coordinates = (150, 300), diameter = 60, character = "a"):
	
	background(255)
	no_stroke()
	dist_from_key = int(dist(circle_coordinates, (mouse_x, mouse_y)))
	#diameter = 0.1 * mouse_y
	text_size(int((0.5)*diameter))
	
	if mouse_is_pressed and (dist_from_key <= diameter/2):
		fill(125)
	else:
		fill(155)
	circle(circle_coordinates, diameter)

	fill(255)
	text(character, circle_coordinates)

	debug(circle_coordinates, dist_from_key)


if __name__ == '__main__':
	run()