import numpy as np

def calcSNR(img2D, img2D_orig):
    # 「ノイズを含む信号のパワー」と「ノイズのみの信号のパワー」を算出
    ## 処理 一行では難しいかも
    img2D = img2D.astype(np.float32)
    img2D_orig = img2D_orig.astype(np.float32)
    S = np.sum(img2D**2)
    N = np.sum((img2D - img2D_orig)**2)

    return 10*np.log10(S/N)

def calcPSNR(img2D, img2D_orig):
    img2D = img2D.astype(np.float32)
    img2D_orig =img2D_orig.astype(np.float32)
    # 信号のピークのパワーを算出
    ## 処理
    h,w = img2D.shape
    P = h*w*255.0**2
    # ノイズ信号のパワーの算出
    ## 処理
    N = np.sum((img2D - img2D_orig)**2)
    # PSNRの算出
    return 10*np.log10(P/N)