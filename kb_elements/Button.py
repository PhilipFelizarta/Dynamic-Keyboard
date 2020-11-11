from p5 import *

#Button object for canvas. Called using, coordinates tuple, diameter, and character
class Button:
	def __init__(self, coordinates, diameter, character, font_name="arial.ttf"):
		self.coordinates = coordinates
		self.diameter = diameter
		self.character = character
		self.font_name = font_name
		arial = create_font(self.font_name, 10)
		text_font(arial)
		text_align(CENTER, CENTER)

	# Calculates distance between button and mouse position, returns a float
	def getDistFromKey(self):
		self.dist_from_key = dist(self.coordinates, (mouse_x, mouse_y))
		return self.dist_from_key

	# Sets the diameter of the button
	def setDiameter(self, diameter):
		self.diameter = diameter

	# Sets a new position for the button
	def setPosition(self, coordinates):
		self.coordinates = coordinates

	# Return the position of the button as a tuple
	def getPosition(self):
		return self.coordinates

	# Returns the diameter of a button, usually a float
	def getDiameter(self):
		return self.diameter

	# Returns the character of the button
	def getCharacter(self):
		return self.character

	# Sets a new character for the button (used for capitalization)
	def setCharacter(self, character):
		self.character = character

	# Sets the character to uppercase.
	def setUpperCase(self, character):
		upper_char = character.upper()
		setCharacter(upper_char)

	# Draws the button on the screen. Put this in the draw() method
	def drawButton(self):
		no_stroke()
		text_size(int((0.5)*self.diameter))

		# If the mouse button is held down, fill with a lighter color
		#TODO: Add text implementation
		if mouse_is_pressed and (self.getDistFromKey() <= self.diameter/2):
			fill(125)
		else:
			fill(155)
		circle(self.coordinates, self.diameter)

		# White text
		fill(255)
		text(self.character, self.coordinates)