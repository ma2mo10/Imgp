# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

def ContrastAdj(img2D, a, b):
    ##処理
    Imax = img2D.max()
    Imin = img2D.min()
    return (b-a) / (Imax-Imin) * (img2D-Imin) + a

def FlatHist(img2D):
    hist = cv2.calcHist([img2D], [0], None, [256], [0,256])
    cdf = hist.cumsum()
    cdf = np.uint8(ContrastAdj(cdf, 0, 255))

    return cdf[img2D]


def main():
    img = cv2.imread('./lena.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # contrastAdjのテスト用
    ##処理
    res = FlatHist(img_gray)

    plt.subplot(2, 2, 1), plt.imshow(img_gray, 'gray', vmin=0, vmax=255)
    plt.subplot(2, 2, 2), plt.imshow(res,'gray',vmin=0,vmax=255)
    plt.subplot(2, 2, 3), plt.hist(img_gray.flatten(), 256,[0, 256])
    plt.subplot(2, 2, 4), plt.hist(res.flatten(), 256, [0, 256])
    plt.show()

if __name__=='__main__':
    main()
