from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from winsound import *
from PIL import Image, ImageTk
from ocr import ocr
from gui import gui
from tkinter import *
import tkinter as tk
from PIL import ImageTk,Image
from script import script
from generate_audio import gtts
from pdf_to_text import pdf
import time
class Root(Tk):

    def __init__(self):
        super(Root, self).__init__()
        self.title("Python Tkinter Dialog Widget")
        self.minsize(640, 400)

        self.labelFrame = ttk.LabelFrame(self, text = "Open File")
        self.labelFrame.grid(column = 0, row = 1, padx = 20, pady = 20)
        self.text=""

        self.emotion=""
        self.L1=None
        self.button()
        self.button2()

    def button(self):
        self.button = ttk.Button(self.labelFrame, text = "Browse A File",command = self.fileDialog)
        self.button.grid(column = 1, row = 1)

    def new_win(self):
        root.destroy()
        top = tk.Tk()
        top.configure(bg='white')
        top.title("EMO TEXT")
        self.L1 = Label(top, text="Enter a sentence", compound=CENTER, fg="#4f54d2", bg="white", font=("Times", 20)).place(
            x=370, y=20)
        E1 = Text(top, width=20, font=('courier', 15, 'bold'))

        E1.place(x=310, y=100)
        E1.insert(INSERT,self.text)
        def retrieve_input():

            input = E1.get("1.0", 'end-1c')
            print(input)
            split_input=input.split(',')
            for index,i in enumerate(split_input):
                self.text = i
                s = script()
                msg = s.get_prediction(i)
                L2 = Label(top, text="", compound=CENTER, fg="#800000", bg="white",
                           font=("Times", 20)).place(x=370, y=50)
                self.emotion = msg
                L2 = Label(top, text="emotion is: " + self.emotion, compound=CENTER, fg="#800000", bg="white",
                           font=("Times", 20)).place(x=50, y=50*index)
                # L2.set(self.emotion)

                #alert_popup("Success!", msg, i)
                audio = gtts(self.text, self.emotion)
                audio.generate()
                time.sleep(1)





        def alert_popup(title, message, path):
            """Generate a pop-up window for special messages."""
            root = Tk()
            root.title(title)

            w = 400  # popup window width
            h = 200  # popup window height

            sw = root.winfo_screenwidth()
            sh = root.winfo_screenheight()

            x = (sw - w) / 2
            y = (sh - h) / 2
            root.geometry('%dx%d+%d+%d' % (w, h, x, y))

            m = message
            m += '\n'
            m += path
            w = Label(root, text=m, width=120, height=10)
            w.pack()
            b = Button(root, text="OK", command=root.destroy, width=10)
            b.pack()
            mainloop()

        g_btn = PhotoImage(file="img0.png")
        img0 = Button(top, image=g_btn, borderwidth=0, command=retrieve_input).place(x=310, y=400)

        img = Image.open("side_img.jpg")
        img = img.resize((300, 500))
        tkimage = ImageTk.PhotoImage(img)
        tk.Label(top, image=tkimage).place(x=0, y=0)

        top.geometry("600x500")
        top.mainloop()

    def button2(self):
        self.button2 = ttk.Button(self.labelFrame, text = "get the text",command = self.new_win)
        self.button2.grid(column = 4, row = 4)

    def fileDialog(self):

        self.filename = filedialog.askopenfilename(initialdir =  "/", title = "Select A File", filetype =
        (("jpeg files","*.jpg"),("pdf files","*.pdf"),("all files","*.*")))
        self.label = ttk.Label(self.labelFrame, text = "")
        self.label.grid(column = 1, row = 2)
        self.label.configure(text = self.filename)


        if self.filename[-3:] == "pdf":
            obj = pdf(self.filename)
            self.text=obj.text
        else:
            img = Image.open(self.filename)
            photo = ImageTk.PhotoImage(img)

            self.label2 = Label(image=photo)
            self.label2.image = photo
            self.label2.grid(column=3, row=4)
            obj = ocr()
            obj.IMAGE_PATH = self.filename
            self.text = obj.get_text()





root = Root()
root.mainloop()
