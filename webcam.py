import cv2

from pcn import pcn_detect
from models import load_model
from utils import draw_face



if __name__ == '__main__':
    # network detection
    nets = load_model()
    cam = cv2.VideoCapture(0)
    while cam.isOpened():
        ret, img = cam.read()
        faces = pcn_detect(img, nets)
        for face in faces:
            draw_face(img, face)
        cv2.imshow('PCN', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
