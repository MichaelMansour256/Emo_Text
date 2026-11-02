from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from winsound import *
from PIL import Image, ImageTk
from ocr import ocr
from gui import gui
from tkinter import *
import tkinter as tk
from PIL import ImageTk, Image
from script import script
from generate_audio import gtts
from pdf_to_text import pdf
from paths import BUTTON_IMAGE, BACKGROUND_IMAGE, BASE_DIR

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
        # Use Toplevel instead of destroying root and creating a new Tk.
        top = tk.Toplevel(self)
        top.configure(bg="white")
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
                # Non-blocking delay using after instead of sleep
                top.after(1000, lambda: None)





        def alert_popup(title, message, path):
            """Generate a pop-up window for special messages."""
            popup = tk.Toplevel(top)
            popup.title(title)

            w = 400  # popup window width
            h = 200  # popup window height

            sw = popup.winfo_screenwidth()
            sh = popup.winfo_screenheight()

            x = (sw - w) // 2
            y = (sh - h) // 2
            popup.geometry(f"{w}x{h}+{x}+{y}")

            m = message
            m += "\n"
            m += path
            w_label = Label(popup, text=m, width=120, height=10)
            w_label.pack()
            b = Button(popup, text="OK", command=popup.destroy, width=10)
            b.pack()
            popup.transient(top)
            popup.grab_set()
            top.wait_window(popup)

        # Use centralized asset paths with fallback for missing assets.
        try:
            g_btn = PhotoImage(file=str(BUTTON_IMAGE))
            img0 = Button(
                top, image=g_btn, borderwidth=0, command=retrieve_input
            ).place(x=310, y=400)
            top._button_image_ref = g_btn  # keep reference to prevent GC
        except Exception:
            img0 = Button(
                top, text="Play", borderwidth=0, command=retrieve_input
            ).place(x=310, y=400)

        try:
            img = Image.open(str(BACKGROUND_IMAGE))
            img = img.resize((300, 500))
            tkimage = ImageTk.PhotoImage(img)
            tk.Label(top, image=tkimage).place(x=0, y=0)
            top._bg_image_ref = tkimage
        except Exception:
            pass

        top.geometry("600x500")

    def button2(self):
        self.button2 = ttk.Button(self.labelFrame, text = "get the text",command = self.new_win)
        self.button2.grid(column = 4, row = 4)

    def fileDialog(self):

        self.filename = filedialog.askopenfilename(
            initialdir=str(BASE_DIR),
            title="Select A File",
            filetype=(("jpeg files", "*.jpg"), ("pdf files", "*.pdf"), ("all files", "*.*")),
        )
        self.label = ttk.Label(self.labelFrame, text="")
        self.label.grid(column=1, row=2)
        self.label.configure(text=self.filename)

        if not self.filename:
            return

        if self.filename.lower().endswith(".pdf"):
            try:
                obj = pdf(self.filename)
                self.text = obj.text
            except Exception as exc:
                self.alert_popup("PDF Error", str(exc), "")
        else:
            try:
                img = Image.open(self.filename)
                photo = ImageTk.PhotoImage(img)

                self.label2 = Label(image=photo)
                self.label2.image = photo
                self.label2.grid(column=3, row=4)
                obj = ocr()
                obj.IMAGE_PATH = self.filename
                self.text = obj.get_text()
            except Exception as exc:
                self.alert_popup("Image Error", str(exc), "")






    def alert_popup(self, title, message, path):
        """Generate a pop-up window for special messages."""
        popup = tk.Toplevel(self)
        popup.title(title)

        w = 400  # popup window width
        h = 200  # popup window height

        sw = popup.winfo_screenwidth()
        sh = popup.winfo_screenheight()

        x = (sw - w) // 2
        y = (sh - h) // 2
        popup.geometry(f"{w}x{h}+{x}+{y}")

        m = message
        m += "\n"
        m += path
        w_label = Label(popup, text=m, width=120, height=10)
        w_label.pack()
        b = Button(popup, text="OK", command=popup.destroy, width=10)
        b.pack()
        popup.transient(self)
        popup.grab_set()
        self.wait_window(popup)


root = Root()
root.mainloop()
