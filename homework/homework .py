import cv2 #计算机视觉库
import numpy as np  #矩阵和数组计算
import matplotlib.pyplot as plt #静态绘图

#to do list
#读取视频：用 cv2.VideoCapture。
#ROI 提取：裁剪出球所在区域，避免干扰。
#颜色空间转换：cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)。
#阈值分割：用 cv2.inRange() 分别得到红色、蓝色、紫色的掩码。
#像素统计：用 cv2.countNonZero() 统计掩码里像素数量。
#结果判定：哪个颜色像素最多，就说明当前是那个球；如果数量很少，可以判断为“无球”。
#展示结果：用 cv2.putText() 把判定结果写在视频帧上，然后显示。

#init
ROI = (200, 170, 200, 200) #下次用x,y, w, h 好调
VIDEO_PATHS = ["res/output.avi", "res/output1.avi"]
KERNEL = np.ones((5, 5), np.uint8)   #创建一个5x5的方形卷积核 所有元素都是1, 数据类型是uint8
MIN_PIX = 300
NO_BALL = 0.01
#输出的字符颜色
COLOR_TEXT = {
    "null": (200,200,200),
    "red": (0,0,255),
    "blue": (255,0,0),
    "purple": (255,0,255),
}
#inRange的上下界
hsv_red_l = np.array([220, 30, 150])
hsv_red_u = np.array([250, 255, 255])
hsv_blue_l = np.array([142, 80, 80])
hsv_blue_u = np.array([184 , 255 ,255])
hsv_purple_l = np.array([184, 60, 60])
hsv_purple_u = np.array([220, 255, 255])


for path in VIDEO_PATHS:
    # 从output.avi中读取帧
    cam = cv2.VideoCapture(path)
    cnt = 0
    while True:
        ret, frame = cam.read()  # 读取一帧
        frame = cv2.rotate(frame, cv2.ROTATE_180)  # 旋转180度（如果视频是倒的）
        if not ret:  # 没有帧了，退出循环
            break
    
        roi = frame[200:400, 270: 470]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV_FULL)

        mask_r = cv2.inRange(hsv, hsv_red_l, hsv_red_u)
        mask_b = cv2.inRange(hsv, hsv_blue_l, hsv_blue_u)
        mask_p = cv2.inRange(hsv, hsv_purple_l, hsv_purple_u)

        mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, KERNEL)#用腐蚀&&膨胀降噪,卷积核大小是KERNEL
        mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_OPEN, KERNEL)
        mask_p = cv2.morphologyEx(mask_p, cv2.MORPH_OPEN, KERNEL)

        cnt_r = cv2.countNonZero(mask_r)    #非常常用的图像颜色识别方法:数对应掩码图中黑/白粒子的数量
        cnt_b = cv2.countNonZero(mask_b)
        cnt_p = cv2.countNonZero(mask_p)

        label = 'null'
        dic = {'red': cnt_r, 'blue': cnt_b, 'purple':cnt_p}
        check = {k:v for k,v in dic.items()
                if v >= MIN_PIX and v/400 >= NO_BALL}#这个是个字典推导式 从dic中找满足条件的
        if check:
            label = max(dic, key=dic.get)   #max(dic) -> 找最大的键   max(dic, key=dic.get) -> 找值最大的键

        cv2.rectangle(frame, (200,170), (400, 370), (0,255,0), 2) #cv2.rectangle(image, pt1, pt2, color, thickness)
        #image 要处理的图像  pt1 左上角坐标  pt2 右下角坐标  color 矩形的颜色(BGR)  thickness 线条粗细(-1是填充)
        cv2.putText(frame, f"{label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_TEXT[label], 2)#在frame 输出label, 在(20,40), 用这个鬼字体、大小，颜色参考COLOR_TEXT，字体线条粗细为2
        
        #这一步没有什么用   h, s, v = cv2.split(hsv)
        # 显示当前帧
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        if (cv2.waitKey(1) & 0xFF) == 27:  # ESC 退出
            break
        # 在这里对 frame 做处理（比如颜色识别）

cam.release()
cv2.destroyAllWindows()