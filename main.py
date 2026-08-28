import tkinter as tk

from model import CarKnnModel
from ui import CarPredictionUI


def main():
    model = CarKnnModel()
    root = tk.Tk()
    CarPredictionUI(root, model)
    root.mainloop()


if __name__ == "__main__":
    main()
