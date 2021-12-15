#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk, RAISED
from tkinter.ttk import Label
from tkinter import scrolledtext
m1 = None
m2 = None
m3 = None
global_response = ''
text_area = None

def setText(text):
    global text_area
    text_area.delete('1.0', tk.END)
    text_area.insert(tk.INSERT, text)

def change1():
    pass

def change2():
    pass

def change3():
   pass


def inicialize():

    # Creating tkinter main window
    win = tk.Tk()
    win.title("ScrolledText Widget")
    global m1
    m1 = tk.IntVar()
    global m2
    m2 = tk.IntVar()
    global m3
    m3 = tk.IntVar()

    # Title Label
    ttk.Label(win,
              text="ScrolledText Widget Example",
              font=("Times New Roman", 15),
              background='green',
              foreground="white")

    tk.Checkbutton(win, text="Modelo 1", command=change1, variable=m1,
                   onvalue=1, offvalue=0).grid()
    tk.Checkbutton(win, text="Modelo 2", command=change2, variable=m2,
                   onvalue=1, offvalue=0).grid()
    tk.Checkbutton(win, text="Modelo 3", command=change3, variable=m3,
                   onvalue=1, offvalue=0).grid()
    # Creating scrolled text
    # area widget
    global text_area
    text_area = scrolledtext.ScrolledText(win,
                                          wrap=tk.WORD,
                                          width=60,
                                          height=30,
                                          font=("Times New Roman",
                                                15))

    text_area.grid(column=0, pady=10, padx=10)

    # Inserting Text which is read only
    text_area.insert(tk.INSERT,global_response)

    # Placing cursor in the text area
    text_area.focus()
    win.mainloop()