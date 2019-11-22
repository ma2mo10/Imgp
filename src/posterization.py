# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
import preprocessing as pre

def postarization(img, val, pos_num):
    # しきい値の設定 =================================================
    T = [63, 127, 191] #[0,,,,255]でも良い # 拡張する人はpos_numを使って設定

    # しきい値を用いてポスタリゼーション ===================================
    ## 同じ大きさの配列を用意（全画素が0で初期化されている）
    pimg = np.zeros(img.shape, dtype=np.uint8)

    ##処理
    pimg[np.logical_and(0<img, img <= T[0])] = val[0] #0
    pimg[np.logical_and(T[0]<img, img<= T[1])] = val[1] #1
    pimg[np.logical_and(T[1]<img, img <= T[2])] = val[2] #2
    pimg[np.logical_and(T[2]<img, img <= 255)] = val[3] #3
    # ポスタリゼーションの結果を出力
    return pimg


def main():
    img = cv2.imread('./lowContrast_lena.png')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ポスタリゼーション後の色数を指定
    pos_num = 4 # 拡張しない場合，必要ない

    # ポスタリゼーション後の値を設定 =====================================
    val = [0, 85, 170, 255] # 拡張する人はpos_numを使って設定
    #img_gray = pre.contrastAdj(img_gray, 0, 255)
    img_gray = pre.FlatHist(img_gray)
    # 均等間隔のポスタリゼーション ======================================
    res = postarization(img_gray.copy(), val, pos_num)

    ## 画像を保存したい場合
    #cv2.imwrite('res.png', res)

    plt.subplot(2, 2, 1), plt.imshow(img_gray, 'gray', vmin=0, vmax=255)
    plt.subplot(2, 2, 2), plt.imshow(res, 'gray', vmin=0, vmax=255)
    plt.subplot(2, 2, 3), plt.hist(img_gray.flatten(), 256, [0, 256])
    plt.subplot(2, 2, 4), plt.hist(res.flatten(), 256, [0, 256])
    plt.show()


if __name__ == '__main__':
    main()
