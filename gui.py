from tkinter import *
import tkinter as tk
from PIL import ImageTk, Image
#import * from script
from script import script
from paths import BUTTON_IMAGE, BACKGROUND_IMAGE
class gui():


    def __init__(self,text):
        self.top=None
        self.E1 = None
        self.g_btn = None
        self.value = None
        self.L1 = None
        self.img=None
        self.img0 =None
        self.tkimage=None
        self.text=text







    def build(self):
        self.top = tk.Tk()
        self.top.geometry("600x500")
        self.top.configure(bg='white')
        self.top.title("EMO TEXT")
        self.L1 = Label(self.top, text="Enter a sentence",compound = CENTER,fg = "#4f54d2",bg="white",font = ("Times",20)).place(x=370,y=20)
        self.value=StringVar(self.top)
        self.E1 = Text(self.top,width=20,font=('courier', 15, 'bold'))
        self.E1.insert(INSERT,self.text)
        self.E1.place(x=310,y=100)
        # Use centralized asset path with fallback for missing assets.
        try:
            g_btn = PhotoImage(file=str(BUTTON_IMAGE))
            self.img0 = Button(
                self.top,
                image=g_btn,
                borderwidth=0,
                command=self.retrieve_input,
            ).place(x=310, y=400)
            self._button_image_ref = g_btn  # keep reference to prevent GC
        except Exception:
            self.img0 = Button(
                self.top,
                text="Play",
                borderwidth=0,
                command=self.retrieve_input,
            ).place(x=310, y=400)

        try:
            self.img = Image.open(str(BACKGROUND_IMAGE))
            self.img = self.img.resize((300, 500))
            self.tkimage = ImageTk.PhotoImage(self.img)
            tk.Label(self.top, image=self.tkimage).place(x=0, y=0)
            self._bg_image_ref = self.tkimage
        except Exception:
            pass  # background image is optional


        self.top.mainloop()

    def retrieve_input(self):
        input = self.E1.get("1.0",'end-1c')
        print(input)
        s=script()
        msg=s.get_prediction(input)
        self.alert_popup("Success!", msg,"")


    def alert_popup(self, title, message, path):
        """Generate a pop-up window for special messages.

        Uses Toplevel (child of the main window) instead of creating a
        second Tk root, which is the correct Tkinter pattern.
        """
        popup = tk.Toplevel(self.top)
        popup.title(title)

        w = 400  # popup window width
        h = 200  # popup window height

        sw = popup.winfo_screenwidth()
        sh = popup.winfo_screenheight()

        x = (sw - w) // 2
        y = (sh - h) // 2
        popup.geometry(f"{w}x{h}+{x}+{y}")

        m = message
        if path:
            m += "\n" + path

        Label(popup, text=m, width=120, height=10).pack()
        Button(popup, text="OK", command=popup.destroy, width=10).pack()
        popup.transient(self.top)
        popup.grab_set()
        self.top.wait_window(popup)



#ff=gui('Tomorrow, and tomorrow, and tomorrow; creeps in this petty pace from day to until the last syil- able of recorded time_ And all our yesterdays have jighted fools the way to dusty day_ ')
#ff.build()
