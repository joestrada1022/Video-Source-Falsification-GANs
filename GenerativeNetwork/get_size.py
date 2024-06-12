from tkinter import filedialog as fd
import cv2

path = fd.askopenfilename()

img = cv2.imread(path)

height, width = img.shape[:2]

print(f'Width: {width}, Height: {height}')