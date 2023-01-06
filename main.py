import dlib
import cv2
import numpy as np
import math

rimg = 'C:\\Users\\Bang\\Desktop\\VIS\\20.jpg'
newimg = 'C:\\Users\\Bang\\Desktop\\VIS\\Result\\20'
Milti=1.2

predictor_path = "D:\\GAME\\shape_predictor_68_face_landmarks.dat\\shape_predictor_68_face_landmarks.dat"
# 使用dlib自带的frontal_face_detector作为我们的特征提取器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def landmark_dec_dlib_fun(img_src):
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    land_marks = []
    rects = detector(img_gray, 0)
    for i in range(len(rects)): 
        land_marks_node = np.matrix([[p.x, p.y] for p in predictor(img_gray, rects[i]).parts()]) 
        land_marks.append(land_marks_node)
    return land_marks


def localTranslationWarp(srcImg, startX, startY, endX, endY, radius):
    ddradius = float(radius * radius)
    copyImg = np.zeros(srcImg.shape, np.uint8)
    copyImg = srcImg.copy()
 # 计算公式中的|m-c|^2
    ddmc = (endX - startX) * (endX - startX) + (endY - startY) * (endY - startY)
    H, W, C = srcImg.shape
    for i in range(W):
        for j in range(H):
 # 计算该点是否在形变圆的范围之内
 # 优化，第一步，直接判断是会在（startX,startY)的矩阵框中
            if math.fabs(i - startX) > radius and math.fabs(j - startY) > radius:
                continue
            distance = (i - startX) * (i - startX) + (j - startY) * (j - startY)
            if (distance < ddradius):
 # 计算出（i,j）坐标的原坐标
 # 计算公式中右边平方号里的部分
                ratio = (ddradius - distance) / (ddradius - distance + ddmc)
                ratio = ratio * ratio
 # 映射原位置
                UX = i - ratio * (endX - startX)
                UY = j - ratio * (endY - startY)
 # 根据双线性插值法得到UX，UY的值
                value = BilinearInsert(srcImg, UX, UY)
 # 改变当前 i ，j的值
                copyImg[j, i] = value
    return copyImg
# 双线性插值法
def BilinearInsert(src, ux, uy):
    w, h, c = src.shape
    if c == 3:
        x1 = int(ux)
        x2 = x1 + 1
        y1 = int(uy)
        y2 = y1 + 1
        part1 = src[y1, x1].astype(np.float) * (float(x2) - ux) * (float(y2) - uy)
        part2 = src[y1, x2].astype(np.float) * (ux - float(x1)) * (float(y2) - uy)
        part3 = src[y2, x1].astype(np.float) * (float(x2) - ux) * (uy - float(y1))
        part4 = src[y2, x2].astype(np.float) * (ux - float(x1)) * (uy - float(y1))
        insertValue = part1 + part2 + part3 + part4
        return insertValue.astype(np.int8)

def face_thin_auto(src):
    landmarks = landmark_dec_dlib_fun(src)
 # 如果未检测到人脸关键点，就不进行瘦脸
    if len(landmarks) == 0:
        return
    thin_image = src
    landmarks_node = landmarks[0]
    endPt = landmarks_node[16]
    for index in range(3, 14, 2):
        start_landmark = landmarks_node[index]
        end_landmark = landmarks_node[index + 2]
        r = math.sqrt((start_landmark[0, 0] - end_landmark[0, 0]) * (start_landmark[0, 0] - end_landmark[0, 0]) +
                        (start_landmark[0, 1] - end_landmark[0, 1]) * (start_landmark[0, 1] - end_landmark[0, 1]))
        thin_image = localTranslationWarp(thin_image, start_landmark[0, 0], start_landmark[0, 1], endPt[0, 0], endPt[0, 1],r*Milti)
 # 显示
    cv2.imshow('thin', thin_image)
    cv2.imwrite(newimg + "-Liq.jpg", thin_image)

def main():
    src = cv2.imread(rimg)
    cv2.imshow('src', src)
    face_thin_auto(src)
    if mode == "1":
        cv2.waitKey(0)

mode = input("Select Mode : 1 = Liquify ; 2 = Beautify")


if mode == "1" or mode == "both" :
    main()

if mode == "2" or mode == "both"  :
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    from copy import deepcopy

    def t2s(img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    img = cv2.imread(rimg)
    #---------------------------------------#
    #转换到hsv域获得对应的二值分割图
    #---------------------------------------#
    def get_face(image_origin,image):
        result_face = deepcopy(image)
        hsv = cv2.cvtColor(image_origin, cv2.COLOR_BGR2HSV) # 把图像转换到HSV色域
        (_h, _s, _v) = cv2.split(hsv) # 图像分割, 分别获取h, s, v 通道分量图像
        skin3 = np.zeros(_h.shape, dtype=np.float32) # 根据源图像的大小创建一个全0的矩阵,用于保存图像数据
        (height,width) = _h.shape # 获取源图像数据的长和宽
    # 遍历图像, 判断HSV通道的数值, 如果在指定范围中, 则置把新图像的点设为255,否则设为0
        for i in range(0, height):
            for j in range(0, width):
                if (_h[i][j] > 5) and (_h[i][j] < 120) and (_s[i][j] > 18) and (_s[i][j] < 255) and (_v[i][j] > 50) and (_v[i][j] < 255):
                    skin3[i][j] = 1.0
                else:
                    skin3[i][j] = 0.0
                    result_face[i,j] = image_origin[i,j]
        result_face = result_face.astype(np.uint8)
        return skin3,result_face
    #--------------------------------------#
    #使用双边滤波获得磨皮效果图
    #--------------------------------------#
    d_20 = cv2.bilateralFilter(img,10,20,20)
    d_100 = cv2.bilateralFilter(img,10,50,50)
    d_200 = cv2.bilateralFilter(img,10,100,100)
    #--------------------------------------# 
    #获得对人脸有效区域进行滤除之后的人脸美颜结果
    #--------------------------------------#
    _, d_20 = get_face(img,d_20)
    _,d_100 = get_face(img,d_100)
    _,d_200 = get_face(img,d_200)
    temp = cv2.hconcat([img,d_20,d_100,d_200])
    temp = deepcopy(temp)
    plt.figure(figsize=(20,20))
    plt.imshow(t2s(temp))
    plt.show()
    cv2.imwrite(newimg + "-Beau.jpg",temp)



