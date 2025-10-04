import cv2
import numpy as np

VIDEO_PATHS = ["res/output.avi", "res/output1.avi"]
WINDOW_NAME = "R2 Ball State"

# 固定 ROI（x, y, w, h）
ROI = (200, 170, 200, 200)

# HSV_FULL 下的颜色阈值（H,S,V ∈ [0,255]）
red_l  = np.array([220,  30, 150], dtype=np.uint8)
red_u  = np.array([250, 255, 255], dtype=np.uint8)

blue_l = np.array([142,  80,  80], dtype=np.uint8)
blue_u = np.array([184, 255, 255], dtype=np.uint8)

purp_l = np.array([184,  60,  60], dtype=np.uint8)
purp_u = np.array([220, 255, 255], dtype=np.uint8)

# 简单去噪用的卷积核
KERNEL = np.ones((5, 5), np.uint8)

# 判定阈值
MIN_PIX = 300
NO_BALL_RATIO = 0.01

COLOR_TEXT = {
    "null": (200,200,200),
    "red": (0,0,255),
    "blue": (255,0,0),
    "purple": (255,0,255),
}

for path in VIDEO_PATHS:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[!] 无法打开：{path}")
        continue

    x, y, w, h = ROI
    roi_area = w * h

    while True:
        ok, frame = cap.read()
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        if not ok:
            break

        # roi 区域
        roi = frame[y:y+h, x:x+w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV_FULL)

        # 掩码
        mask_r = cv2.inRange(hsv, red_l,  red_u)
        mask_b = cv2.inRange(hsv, blue_l, blue_u)
        mask_p = cv2.inRange(hsv, purp_l, purp_u)

        # 去噪：开运算 = 腐蚀 + 膨胀
        mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, KERNEL)
        mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_OPEN, KERNEL)
        mask_p = cv2.morphologyEx(mask_p, cv2.MORPH_OPEN, KERNEL)

        # 计数
        cnt_r = cv2.countNonZero(mask_r)
        cnt_b = cv2.countNonZero(mask_b)
        cnt_p = cv2.countNonZero(mask_p)

        # 判定
        label = "null"
        cand = {"red": cnt_r, "blue": cnt_b, "purple": cnt_p}
        cand = {k:v for k,v in cand.items()
                if v >= MIN_PIX and v/roi_area >= NO_BALL_RATIO}
        if cand:
            label = max(cand, key=cand.get)

        # 显示
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)    #cv2.rectangle(image, pt1, pt2, color, thickness)
        #image 要处理的图像  pt1 左上角坐标  pt2 右下角坐标  color 矩形的颜色(BGR)  thickness 线条粗细(-1是填充)
        cv2.putText(frame, f"State:{label}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_TEXT[label], 2)
        cv2.imshow(WINDOW_NAME, frame)

        if (cv2.waitKey(1) & 0xFF) == 27:  # ESC 退出
            break

    cap.release()

cv2.destroyAllWindows()




