import easyocr
class ocr():

  def __init__(self):
    self.IMAGE_PATH = ''

  def get_text(self):
    reader = easyocr.Reader(['en'],gpu=False)
    result = reader.readtext(self.IMAGE_PATH)
    s=""
    for i in range(len(result)):
      s+=result[i][1]+" "
    return s