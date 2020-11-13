from Button import *

class SpecialButton(Button):
  def __init__(self, coordinates, diameter, character):
    super().__init__(coordinates, diameter, character, font_name="arial.ttf")

  def spaceKey(text_string):
    if(self.getDistFromKey() <= self.diameter/2):
      text_string.append(' ')

  def backspaceKey(text_string):
    if(self.getDistFromKey() <= self.diameter/2):
      del text_string[-1]

  def capsKey():
    pass

  def returnKey(text_string):
    if(self.getDistFromKey() <= self.diameter/2):
      print("".join(text_string))
      del text_string[:]