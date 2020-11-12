from p5 import *

class Display:
  def __init__(self, coordinates):
    self.coordinates = coordinates
    self.font_name = "arial.ttf"
    arial = create_font(self.font_name, 15)
    text_font(arial)
    text_align("LEFT", "BOTTOM")
    #text_size(10)

  def drawTextBox(self, size):
    fill(255)
    stroke(0)
    rect(self.coordinates[0], self.coordinates[1], size[0], size[1])

  def drawText(self, text_string, coordinates):
    display_length = len(text_string) - 31
    if(display_length > 0):
      self.text_string = text_string[display_length:-1]
    else:
      self.text_string = text_string
    fill(0)
    text(self.text_string, coordinates)