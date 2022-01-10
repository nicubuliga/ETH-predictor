from PIL import ImageTk, Image

import rnn
import tkinter as tk
from tkinter import ttk


def start_pressed(root, durate, dataset_type):
    root.withdraw()
    durate_value = int(durate.get())

    if durate_value > 1000:
        durate_value = 1000
    rnn.lstm_model(dataset_type.get(), durate_value)
    root.deiconify()


def initialize_menu():
    root = tk.Tk()
    root.title('ETH-Predictor')
    root.geometry('640x427')
    root.config(bg='yellow')

    img = ImageTk.PhotoImage(Image.open("back2.jpg"))

    # Show image using label
    background= tk.Label(root, image=img)
    background.place(x=0, y=0)

    # label declaration
    title_label = ttk.Label(root, text="ETH-Predictor", foreground="black",background='white',
                            font=("Comic Sans MS", 30))
    title_label.place(x=150, y=0)

    dataset_type_label = ttk.Label(root, text="Dataset type", foreground="black",background='white',
                                   font=("Comic Sans MS", 15))
    dataset_type_label.place(x=150, y=150)

    durate_label = ttk.Label(root, text="Durate:", foreground="black",background="white",
                             font=("Comic Sans MS", 15))
    durate_label.place(x=150, y=180)

    # comboBox
    dataset_type_combo = ttk.Combobox(root, width=10, font=("Comic Sans MS", 10))
    dataset_type_combo['values'] = ('daily', 'hourly', 'minute')
    dataset_type_combo.set('daily')
    dataset_type_combo.place(x=310, y=158)
    dataset_type_combo.state(['readonly'])

    durate_spinbox = tk.Spinbox(root, from_=1, to=1000, width=5, font=("Comic Sans MS", 10),
                                highlightbackground="white")
    durate_spinbox.place(x=310, y=188)

    start_button = tk.Button(root, text="Start", font=("Comic Sans MS", 20))
    start_button['command'] = lambda: start_pressed(root, durate_spinbox, dataset_type_combo)
    start_button.place(x=230, y=230)

    root.mainloop()


if __name__ == "__main__":
    initialize_menu()
