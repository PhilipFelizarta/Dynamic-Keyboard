from p5 import *

class Display:
  def __init__(self, coordinates):
    self.coordinates = coordinates
    self.text_string = text_string
    self.font_name = font_name
    text_align("LEFT", "BOTTOM")

  def drawTextBox(self, coordinates, size):
    fill(255)
    stroke(0)
    rect(coordinates[0], coordinates[1], size[0], size[1])

  def drawText(self, text_string, coordinates):
    self.text_string = text_string
    fill(0)
    text(text_string, coordinates)