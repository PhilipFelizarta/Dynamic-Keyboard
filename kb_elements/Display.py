from p5 import *

class Display:
  def __init__(self, coordinates):
    self.coordinates = coordinates
    text_align("LEFT", "BOTTOM")

  def drawTextBox(self, size):
    fill(255)
    stroke(0)
    rect(self.coordinates[0], self.coordinates[1], size[0], size[1])

  def drawText(self, text_string, coordinates):
    self.text_string = text_string
    fill(0)
    text(self.text_string, coordinates)