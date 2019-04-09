import cv2

import pcn


if __name__ == '__main__':
    # network detection
    cam = cv2.VideoCapture(0)
    while cam.isOpened():
        ret, img = cam.read()
        winlist = pcn.detect(img)
        img = pcn.draw(img, winlist)
        cv2.imshow('PCN', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
