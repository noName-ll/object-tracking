import cv2
import numpy as np


def cutout(archive, obj, angle, side):
    raw = archive.read(f"coil-100/obj{obj}__{angle}.png")
    im = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    mask = im.max(axis=2) > 18
    x, y, w, h = cv2.boundingRect(mask.astype('uint8'))
    shape = (max(1,round(side*w/max(w,h))), max(1,round(side*h/max(w,h))))
    return (cv2.resize(im[y:y+h,x:x+w], shape),
            cv2.resize(mask[y:y+h,x:x+w].astype('uint8'), shape, interpolation=cv2.INTER_NEAREST).astype(bool))


def paste(archive, frame, obj, angle, side, x, y):
    patch, mask = cutout(archive, obj, angle, side)
    h,w = mask.shape
    frame[y:y+h,x:x+w][mask] = patch[mask]
    return np.array([x,y,w,h],float)


def iou(a,b):
    if a is None or b is None:
        return 0
    a,b = np.asarray(a),np.asarray(b)
    intersection = np.maximum(0,np.minimum(a[:2]+a[2:],b[:2]+b[2:])-np.maximum(a[:2],b[:2])).prod()
    return intersection/(a[2:].prod()+b[2:].prod()-intersection)
