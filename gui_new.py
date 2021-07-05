from tkinter import *
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk,Image
#import * from script
from script import script
class gui(tk):
    def __init__(self):
        super(gui, self).__init__()
        self.title("Python Tkinter Dialog Widget")
        self.minsize(640, 400)





    top = tk.Tk()
    top.configure(bg='white')
    top.title("EMO TEXT")
    L1 = Label(top, text="Enter a sentence",compound = CENTER,fg = "#4f54d2",bg="white",font = ("Times",20)).place(x=370,y=20)
    value=StringVar(top)
    E1 = Text(top,width=20,font=('courier', 15, 'bold'))
    E1.place(x=310,y=100)

    def retrieve_input(self):
        input = self.E1.get("1.0",'end-1c')
        print(input)
        s=script()
        msg=s.get_prediction(input)
        self.alert_popup("Success!", msg,"")


    def alert_popup(self,title, message, path):
        """Generate a pop-up window for special messages."""
        root = Tk()
        root.title(title)

        w = 400     # popup window width
        h = 200     # popup window height

        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()

        x = (sw - w)/2
        y = (sh - h)/2
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
    img0 = Button(top, image=g_btn,borderwidth=0,command=retrieve_input).place(x=310,y=400)

    img = Image.open("side_img.jpg")
    img = img.resize((300, 500))
    tkimage = ImageTk.PhotoImage(img)
    tk.Label(top, image=tkimage).place(x=0,y=0)

    top.geometry("600x500")
    top.mainloop()


root=gui()
root.mainloop()
