from PIL import Image
import numpy as np
import cv2

def RGB_to_stdGray(image):
    # 传入一张RGB图象的np.array，转换为相同大小的灰度图
    # 考虑到伽马矫正的效果很差，暂时没有做这一步骤
    # 输入：高*宽*通道数(3)
    # 输出：高*宽
    (b, g, r) = cv2.split(image)
    gray = 0.114 * b + 0.587 * g + 0.299 * r
    gray = gray.astype(np.float32)
    # 下面两行可以测试灰度函数的有效性
    # outputImg=Image.fromarray(gray*255.0)
    # outputImg.show()
    return gray

def getHOGfeat(image, stride = 8, orientations = 8, pixels_per_cell = (8, 8), cells_per_block = (2, 2)):
    # 传入一张RGBA四通道图象的tensor，转换为提取了HOG特征的矩阵
    # 输入：高*宽*通道数(4)
    # 输出：n_blocksx, n_blocksy, bx * by * orientations
    # 输出默认情况下为14*14*32
    image = RGB_to_stdGray(image) # 转换为标准大小灰度图
    cx, cy = pixels_per_cell
    bx, by = cells_per_block
    sx, sy = image.shape
    n_cellsx = sx // cx # x方向上cell的个数
    n_cellsy = sy // cy # y方向上cell的个数
    n_blocksx = (n_cellsx - bx) * cx // stride # x方向上block的个数
    n_blocksy = (n_cellsy - by) * cy //stride # y方向上block的个数
    gx = np.zeros((sx, sy), dtype = np.float32) # x方向上的梯度矩阵
    gy = np.zeros((sx, sy), dtype = np.float32) # y方向上的梯度矩阵
    eps = 1e-5
    grad = np.zeros((sx, sy, 2), dtype = np.float32)
   
    # 获得梯度矩阵
    for i in range(1, sx-1):
        for j in range(1, sy-1):
            gx[i, j] = image[i, j-1] - image[i, j+1]
            gy[i, j] = image[i+1, j] - image[i-1, j]
            grad[i, j, 0] = np.arctan(gy[i,j] / (gx[i, j] + eps)) * 180 / np.pi
            if gx[i, j] < 0:
                grad[i, j, 0] += 180
            grad[i, j, 0] = (grad[i, j, 0] + 360) % 360
            grad[i, j, 1] = np.sqrt(gy[i, j] ** 2 + gx[i, j] ** 2)
    
    # 将梯度矩阵标准化
    # 待调整！！
    normalised_blocks = np.zeros((n_blocksx, n_blocksy, bx * by * orientations))
    for y in range(n_blocksy):
        for x in range(n_blocksx):
            block = grad[y*stride : y*stride+cy*by, x*stride : x*stride+cx*bx]
            hist_block = np.zeros(32, dtype = np.float32)
            for k in range(by):
                for m in range(bx):
                    cell = block[k*cy : (k+1)*cy, m*cx :(m+1)*cx]
                    hist_cell = np.zeros(orientations,dtype = np.float32)
                    for i in range(cy):
                        for j in range(cx):
                            n = int(cell[i,j,0] * orientations // 360)
                            hist_cell[n] += cell[i,j,1]
                    hist_block[(k*bx+m)*orientations:(k*bx+m+1)*orientations] = hist_cell[:]
            normalised_blocks[y, x, :] = hist_block / np.sqrt(hist_block.sum() ** 2 + eps)
    return normalised_blocks
