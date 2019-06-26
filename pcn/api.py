import cv2
import numpy as np

from .models import load_model
from .utils import crop_face, draw_face
from .pcn import pcn_detect


nets = load_model()    

def detect(img):
    if type(img) == str:
        img = cv2.imread(img)
    winlist = pcn_detect(img, nets) 
    return winlist

def crop(img, winlist, size=200):
    """
    Returns:
        list of [face, location] 
    """
    faces = list(map(lambda win: crop_face(img, win, size), winlist))
    return faces

def draw(img, winlist):
    list(map(lambda win: draw_face(img, win), winlist))
    return img

def show(img, is_crop=False):
    img = cv2.imread(img)
    winlist = detect(img)
    if is_crop:
        faces = crop(img, winlist)
        faces = [f[0] for f in faces]
        img = np.hstack(faces)
    else:
        draw(img, winlist)
    cv2.imshow("Show", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

