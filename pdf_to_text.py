import PyPDF2

class pdf():
    def __init__(self,path):
        self.file_path=path
        pdffileobj = open(path, 'rb')
        pdfreader = PyPDF2.PdfFileReader(pdffileobj)
        x = pdfreader.numPages
        print(x)
        pageobj = pdfreader.getPage(x - 1)
        text1 = pageobj.extractText()
        print(text1)
        self.text=text1
        #file1 = open(r"1.txt", "a")
        #file1.writelines(text)

