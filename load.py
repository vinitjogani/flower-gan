import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 as cv


def load(path):
    image = cv.imread(path)
    image = cv.resize(image, (64, 64))
    return image


def view(image, ax=plt):
    image = image.astype('float64')
    image = (image - image.min())/(image.max() - image.min())
    image = (image * 255).astype('uint8')
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    ax.imshow(image)


def load_from_dir(d):
    X = []
    for f in os.listdir(d):
        if not f.endswith(".jpg") and not f.endswith(".png"):
            continue
        X.append(load(f"{d}/{f}"))
    X = np.array(X)
    X = X * 2./256 - 1
    return X
