import os
import numpy as np
import cv2
import torch

from .models import load_model
from .utils import Window, draw_face



# global settings
EPS = 1e-5
minFace_ = 20 * 1.4
scale_ = 1.414
stride_ = 8
classThreshold_ = [0.37, 0.43, 0.97]
nmsThreshold_ = [0.8, 0.8, 0.3]
angleRange_ = 45
stable_ = 0


class Window2:
    def __init__(self, x, y, w, h, angle, scale, conf):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.angle = angle
        self.scale = scale
        self.conf = conf


def preprocess_img(img, dim=None):
    if dim:
        img = cv2.resize(img, (dim, dim), interpolation=cv2.INTER_NEAREST)
    return img - np.array([104, 117, 123])

def resize_img(img, scale:float):
    h, w = img.shape[:2]
    h_, w_ = int(h / scale), int(w / scale)
    img = img.astype(np.float32) # fix opencv type error
    ret = cv2.resize(img, (w_, h_), interpolation=cv2.INTER_NEAREST)
    return ret

def pad_img(img:np.array):
    row = min(int(img.shape[0] * 0.2), 100)
    col = min(int(img.shape[1] * 0.2), 100)
    ret = cv2.copyMakeBorder(img, row, row, col, col, cv2.BORDER_CONSTANT)
    return ret

def legal(x, y, img):
    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
        return True
    else:
        return False

def inside(x, y, rect:Window2):
    if rect.x <= x < (rect.x + rect.w) and rect.y <= y < (rect.y + rect.h):
        return True
    else:
        return False

def smooth_angle(a, b):
    if a > b:
        a, b = b, a
    diff = (b - a) % 360
    if diff < 180:
        return a + diff // 2
    else:
        return b + (360 - diff) // 2

# use global variable `prelist` to mimic static variable in C++
prelist = []
def smooth_window(winlist):
    global prelist
    for win in winlist:
        for pwin in prelist:
            if IoU(win, pwin) > 0.9:
                win.conf = (win.conf + pwin.conf) / 2
                win.x = pwin.x
                win.y = pwin.y
                win.w = pwin.w
                win.h = pwin.h
                win.angle = pwin.angle
            elif IoU(win, pwin) > 0.6:
                win.conf = (win.conf + pwin.conf) / 2
                win.x = (win.x + pwin.x) // 2
                win.y = (win.y + pwin.y) // 2
                win.w = (win.w + pwin.w) // 2
                win.h = (win.h + pwin.h) // 2
                win.angle = smooth_angle(win.angle, pwin.angle)
    prelist = winlist
    return winlist

def IoU(w1:Window2, w2:Window2) -> float:
    xOverlap = max(0, min(w1.x + w1.w - 1, w2.x + w2.w - 1) - max(w1.x, w2.x) + 1)
    yOverlap = max(0, min(w1.y + w1.h - 1, w2.y + w2.h - 1) - max(w1.y, w2.y) + 1)
    intersection = xOverlap * yOverlap
    unio = w1.w * w1.h + w2.w * w2.h - intersection
    return intersection / unio

def NMS(winlist, local:bool, threshold:float):
    length = len(winlist)
    if length == 0:
        return winlist
    winlist.sort(key=lambda x: x.conf, reverse=True)
    flag = [0] * length
    for i in range(length):
        if flag[i]:
            continue
        for j in range(i+1, length):
            if local and abs(winlist[i].scale - winlist[j].scale) > EPS:
                continue
            if IoU(winlist[i], winlist[j]) > threshold:
                flag[j] = 1
    ret = [winlist[i] for i in range(length) if not flag[i]]
    return ret

def deleteFP(winlist):
    length = len(winlist)
    if length == 0:
        return winlist
    winlist.sort(key=lambda x: x.conf, reverse=True)
    flag = [0] * length
    for i in range(length):
        if flag[i]:
            continue
        for j in range(i+1, length):
            win = winlist[j]
            if inside(win.x, win.y, winlist[i]) and inside(win.x + win.w - 1, win.y + win.h - 1, winlist[i]):
                flag[j] = 1
    ret = [winlist[i] for i in range(length) if not flag[i]]
    return ret


# using if-else to mimic method overload in C++
def set_input(img):
    if type(img) == list:
        img = np.stack(img, axis=0)
    else:
        img = img[np.newaxis, :, :, :]
    img = img.transpose((0, 3, 1, 2))
    return torch.FloatTensor(img)


def trans_window(img, imgPad, winlist):
    """transfer Window2 to Window1 in winlist"""
    row = (imgPad.shape[0] - img.shape[0]) // 2
    col = (imgPad.shape[1] - img.shape[1]) // 2
    ret = list()
    for win in winlist:
        if win.w > 0 and win.h > 0:
            ret.append(Window(win.x-col, win.y-row, win.w, win.angle, win.conf))
    return ret

def stage1(img, imgPad, net, thres):
    row = (imgPad.shape[0] - img.shape[0]) // 2
    col = (imgPad.shape[1] - img.shape[1]) // 2
    winlist = []
    netSize = 24
    curScale = minFace_ / netSize
    img_resized = resize_img(img, curScale)
    while min(img_resized.shape[:2]) >= netSize:
        img_resized = preprocess_img(img_resized)
        # net forward
        net_input = set_input(img_resized)
        with torch.no_grad():
            net.eval()
            cls_prob, rotate, bbox = net(net_input)

        w = netSize * curScale
        for i in range(cls_prob.shape[2]): # cls_prob[2]->height
            for j in range(cls_prob.shape[3]): # cls_prob[3]->width
                if cls_prob[0, 1, i, j].item() > thres:
                    sn = bbox[0, 0, i, j].item()
                    xn = bbox[0, 1, i, j].item()
                    yn = bbox[0, 2, i, j].item()
                    rx = int(j * curScale * stride_ - 0.5 * sn * w + sn * xn * w + 0.5 * w) + col
                    ry = int(i * curScale * stride_ - 0.5 * sn * w + sn * yn * w + 0.5 * w) + row
                    rw = int(w * sn)
                    if legal(rx, ry, imgPad) and legal(rx + rw - 1, ry + rw -1, imgPad):
                        if rotate[0, 1, i, j].item() > 0.5:
                            winlist.append(Window2(rx, ry, rw, rw, 0, curScale, cls_prob[0, 1, i, j].item()))
                        else:
                            winlist.append(Window2(rx, ry, rw, rw, 180, curScale, cls_prob[0, 1, i, j].item()))
        img_resized = resize_img(img_resized, scale_)
        curScale = img.shape[0] / img_resized.shape[0]
    return winlist

def stage2(img, img180, net, thres, dim, winlist):
    length = len(winlist)
    if length == 0:
        return winlist
    datalist = []
    height = img.shape[0]
    for win in winlist:
        if abs(win.angle) < EPS:
            datalist.append(preprocess_img(img[win.y:win.y+win.h, win.x:win.x+win.w, :], dim))
        else:
            y2 = win.y + win.h -1
            y = height - 1 - y2
            datalist.append(preprocess_img(img180[y:y+win.h, win.x:win.x+win.w, :], dim))
    # net forward
    net_input = set_input(datalist)
    with torch.no_grad():
        net.eval()
        cls_prob, rotate, bbox = net(net_input)

    ret = []
    for i in range(length):
        if cls_prob[i, 1].item() > thres:
            sn = bbox[i, 0].item()
            xn = bbox[i, 1].item()
            yn = bbox[i, 2].item()
            cropX = winlist[i].x
            cropY = winlist[i].y
            cropW = winlist[i].w
            if abs(winlist[i].angle) > EPS:
                cropY = height - 1 - (cropY + cropW - 1)
            w = int(sn * cropW)
            x = int(cropX - 0.5 * sn * cropW + cropW * sn * xn + 0.5 * cropW)
            y = int(cropY - 0.5 * sn * cropW + cropW * sn * yn + 0.5 * cropW)
            maxRotateScore = 0
            maxRotateIndex = 0
            for j in range(3):
                if rotate[i, j].item() > maxRotateScore:
                    maxRotateScore = rotate[i, j].item()
                    maxRotateIndex = j
            if legal(x, y, img) and legal(x+w-1, y+w-1, img):
                angle = 0
                if abs(winlist[i].angle) < EPS:
                    if maxRotateIndex == 0:
                        angle = 90
                    elif maxRotateIndex == 1:
                        angle = 0
                    else:
                        angle = -90
                    ret.append(Window2(x, y, w, w, angle, winlist[i].scale, cls_prob[i, 1].item()))
                else:
                    if maxRotateIndex == 0:
                        angle = 90
                    elif maxRotateIndex == 1:
                        angle = 180
                    else:
                        angle = -90
                    ret.append(Window2(x, height-1-(y+w-1), w, w, angle, winlist[i].scale, cls_prob[i, 1].item()))
    return ret

def stage3(imgPad, img180, img90, imgNeg90, net, thres, dim, winlist):
    length = len(winlist)
    if length == 0:
        return winlist

    datalist = []
    height, width = imgPad.shape[:2]

    for win in winlist:
        if abs(win.angle) < EPS:
            datalist.append(preprocess_img(imgPad[win.y:win.y+win.h, win.x:win.x+win.w, :], dim))
        elif abs(win.angle - 90) < EPS:
            datalist.append(preprocess_img(img90[win.x:win.x+win.w, win.y:win.y+win.h, :], dim))
        elif abs(win.angle + 90) < EPS:
            x = win.y
            y = width - 1 - (win.x + win.w -1)
            datalist.append(preprocess_img(imgNeg90[y:y+win.h, x:x+win.w, :], dim))
        else:
            y2 = win.y + win.h - 1
            y = height - 1 - y2
            datalist.append(preprocess_img(img180[y:y+win.h, win.x:win.x+win.w], dim))
    # network forward
    net_input = set_input(datalist)
    with torch.no_grad():
        net.eval()
        cls_prob, rotate, bbox = net(net_input)

    ret = []
    for i in range(length):
        if cls_prob[i, 1].item() > thres:
            sn = bbox[i, 0].item()
            xn = bbox[i, 1].item()
            yn = bbox[i, 2].item()
            cropX = winlist[i].x
            cropY = winlist[i].y
            cropW = winlist[i].w
            img_tmp = imgPad
            if abs(winlist[i].angle - 180) < EPS:
                cropY = height - 1 - (cropY + cropW -1)
                img_tmp = img180
            elif abs(winlist[i].angle - 90) < EPS:
                cropX, cropY = cropY, cropX
                img_tmp = img90
            elif abs(winlist[i].angle + 90) < EPS:
                cropX = winlist[i].y
                cropY = width -1 - (winlist[i].x + winlist[i].w - 1)
                img_tmp = imgNeg90

            w = int(sn * cropW)
            x = int(cropX - 0.5 * sn * cropW + cropW * sn * xn + 0.5 * cropW)
            y = int(cropY - 0.5 * sn * cropW + cropW * sn * yn + 0.5 * cropW)
            angle = angleRange_ * rotate[i, 0].item()
            if legal(x, y, img_tmp) and legal(x+w-1, y+w-1, img_tmp):
                if abs(winlist[i].angle) < EPS:
                    ret.append(Window2(x, y, w, w, angle, winlist[i].scale, cls_prob[i, 1].item()))
                elif abs(winlist[i].angle - 180) < EPS:
                    ret.append(Window2(x, height-1-(y+w-1), w, w, 180-angle, winlist[i].scale, cls_prob[i, 1].item()))
                elif abs(winlist[i].angle - 90) < EPS:
                    ret.append(Window2(y, x, w, w, 90-angle, winlist[i].scale, cls_prob[i, 1].item()))
                else:
                    ret.append(Window2(width-y-w, x, w, w, -90+angle, winlist[i].scale, cls_prob[i, 1].item()))
    return ret

def detect(img, imgPad, nets):
    img180 = cv2.flip(imgPad, 0)
    img90 = cv2.transpose(imgPad)
    imgNeg90 = cv2.flip(img90, 0)

    winlist = stage1(img, imgPad, nets[0], classThreshold_[0])
    winlist = NMS(winlist, True, nmsThreshold_[0])
    winlist = stage2(imgPad, img180, nets[1], classThreshold_[1], 24, winlist)
    winlist = NMS(winlist, True, nmsThreshold_[1])
    winlist = stage3(imgPad, img180, img90, imgNeg90, nets[2], classThreshold_[2], 48, winlist)
    winlist = NMS(winlist, False, nmsThreshold_[2])
    winlist = deleteFP(winlist)
    return winlist

def pcn_detect(img, nets):
    imgPad = pad_img(img)
    winlist = detect(img, imgPad, nets)
    if stable_:
        winlist = smooth_window(winlist)
    return trans_window(img, imgPad, winlist)


if __name__ == '__main__':
    # usage settings
    import sys
    if len(sys.argv) != 2:
        print("Usage: python3 pcn.py path/to/img")
        sys.exit()
    else:
        imgpath = sys.argv[1]
    # network detection
    nets = load_model()
    img = cv2.imread(imgpath)
    faces = pcn_detect(img, nets)
    # draw image
    for face in faces:
        draw_face(img, face)
    # show image
    cv2.imshow("pytorch-PCN", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # save image
    name = os.path.basename(imgpath)
    cv2.imwrite('result/ret_{}'.format(name), img)