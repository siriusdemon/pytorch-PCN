"code originally in PCN.h"
import numpy as np
import cv2



class Window:
    def __init__(self, x, y, width, angle, score):
        self.x = x
        self.y = y
        self.width = width
        self.angle = angle
        self.score = score

def rotate_point(x, y, centerX, centerY, angle):
    x -= centerX
    y -= centerY
    theta = -angle * np.pi / 180
    rx = int(centerX + x * np.cos(theta) - y * np.sin(theta))
    ry = int(centerY + x * np.sin(theta) + y * np.cos(theta))
    return rx, ry

def draw_line(img, pointlist):
    thick = 2
    cyan = (255, 255, 0)
    blue = (255, 0, 0)
    cv2.line(img, pointlist[0], pointlist[1], cyan, thick)
    cv2.line(img, pointlist[1], pointlist[2], cyan, thick)
    cv2.line(img, pointlist[2], pointlist[3], cyan, thick)
    cv2.line(img, pointlist[3], pointlist[0], blue, thick)

def draw_face(img, face:Window):
    x1 = face.x
    y1 = face.y
    x2 = face.width + face.x -1
    y2 = face.width + face.y -1
    centerX = (x1 + x2) // 2
    centerY = (y1 + y2) // 2
    lst = (x1, y1), (x1, y2), (x2, y2), (x2, y1)
    pointlist = [rotate_point(x, y, centerX, centerY, face.angle) for x, y in lst]
    draw_line(img, pointlist)


def crop_face(img, face:Window, crop_size=200):
    x1 = face.x
    y1 = face.y
    x2 = face.width + face.x - 1
    y2 = face.width + face.y - 1
    centerX = (x1 + x2) // 2
    centerY = (y1 + y2) // 2
    srcTriangle = np.array([
        rotate_point(x1, y1, centerX, centerY, face.angle),
        rotate_point(x1, y2, centerX, centerY, face.angle),
        rotate_point(x2, y2, centerX, centerY, face.angle),
    ], dtype=np.float32)
    dstTriangle = np.array([
        (0, 0),
        (0, crop_size - 1),
        (crop_size - 1, crop_size - 1),
    ], dtype=np.float32)
    rotMat = cv2.getAffineTransform(srcTriangle, dstTriangle)
    ret = cv2.warpAffine(img, rotMat, (crop_size, crop_size))
    return ret
