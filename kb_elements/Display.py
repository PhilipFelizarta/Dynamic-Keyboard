from p5 import *

class Display:
  def __init__(self, coordinates, text_string, font_name="arial.ttf"):
    self.coordinates = coordinates
    self.text_string = text_string
    self.font_name = font_name
    #arial = create_font(self.font_name, 10)
    #text_font(arial)
    #text_align(LEFT, CENTER)

  def add_string(self, text_string):
    text(text_string)

  def drawBox(self, coordinates, size):
    fill(255)
    stroke(0)
    rect(coordinates[0], coordinates[1], size[0], size[1])