from gtts import gTTS
from IPython.display import Audio
import os

class gtts():
  def __init__(self,text,emotion):
    self.story_text=text
    self.list_text = self.story_text.split('.')
    self.emotion=emotion
  def generate(self):
    for i in range(len(self.list_text)):
      directory = self.emotion
      print(i)
      tts = gTTS(self.list_text[i])
      mp3name = str(i)
      tts.save("%s.wav" % os.path.join("gtts/"+directory, mp3name))