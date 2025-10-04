import cv2
import numpy as np
from collections import deque

# ========== 1) 可调参数 ==========
VIDEO_PATHS = ["res/output.avi", "output1.avi"]  # 两个视频依次处理
SAVE_VIDEO   = False                              # 是否保存结果视频
SHOW_MASKS   = False                              # 是否展示三色掩码窗口
WINDOW_NAME  = "R2 Ball State"

# ROI：按你的画面定一个矩形区域 (x, y, w, h)
# 提示：先把 SHOW_MASKS=True，用鼠标看坐标或直接 print(frame.shape) 再微调
ROI = (200, 270, 200, 200)   # 示例：从(400,220)开始的 200x200 区域

# HSV 颜色阈值（OpenCV 的 H ∈ [0,180]）
# 注意：红色有“跨 0 度”问题，常用双区间
HSV_RED_1 = ([0,   90, 90], [10, 255, 255])      # 低红
HSV_RED_2 = ([170, 90, 90], [180,255, 255])      # 高红
HSV_BLUE  = ([100, 80, 80], [130,255,255])       # 蓝
HSV_PURPLE= ([130, 60, 60], [155,255,255])       # 紫（视画面而定，可与蓝靠近）

# 形态学 & 噪声控制
KERNEL    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
MIN_PIX   = 300          # 认为“检测到该颜色”的最小像素数（按 ROI 面积调）
NO_BALL_RATIO = 0.01     # 掩码像素/ROI像素 低于此比例视为无球(作二次保险)

# 多帧平滑：用最近 N 帧投票，降低抖动
SMOOTH_N  = 5
LABELS    = ["无球", "红球", "蓝球", "紫球"]
COLORS_BRG= {
    "无球": (200,200,200),
    "红球": (0,0,255),
    "蓝球": (255,0,0),
    "紫球": (255,0,255),
}

# ========== 2) 工具函数 ==========
def make_mask(hsv, lower, upper):
    lower = np.array(lower, dtype=np.uint8)
    upper = np.array(upper, dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)

def red_mask(hsv):
    m1 = make_mask(hsv, *HSV_RED_1)
    m2 = make_mask(hsv, *HSV_RED_2)
    return cv2.bitwise_or(m1, m2)

def postprocess_mask(mask):
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=1)
    return mask

def dominant_color_count(hsv_roi):
    # 计算三色像素数量
    m_red    = postprocess_mask(red_mask(hsv_roi))
    m_blue   = postprocess_mask(make_mask(hsv_roi, *HSV_BLUE))
    m_purple = postprocess_mask(make_mask(hsv_roi, *HSV_PURPLE))

    c_red    = int(cv2.countNonZero(m_red))
    c_blue   = int(cv2.countNonZero(m_blue))
    c_purple = int(cv2.countNonZero(m_purple))

    return (c_red, c_blue, c_purple), (m_red, m_blue, m_purple)

def decide_label(counts, roi_area):
    c_red, c_blue, c_purple = counts
    color_counts = {"红球": c_red, "蓝球": c_blue, "紫球": c_purple}

    # 先做最小阈值与比例阈值过滤（避免微小噪点）
    filtered = {k: v for k, v in color_counts.items()
                if v >= MIN_PIX and v/roi_area >= NO_BALL_RATIO}

    if not filtered:
        return "无球"

    # 取像素最多的那个颜色
    return max(filtered, key=filtered.get)

def draw_overlay(frame, roi, label, counts):
    x,y,w,h = roi
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    text = f"State: {label} | R:{counts[0]} B:{counts[1]} P:{counts[2]}"
    cv2.putText(frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLORS_BRG[label], 2, cv2.LINE_AA)
    return frame

def smooth_vote(vote_q, new_label):
    vote_q.append(new_label)
    # 多帧投票
    best = max(set(vote_q), key=vote_q.count)
    return best

# ========== 3) 主处理流程 ==========
def process_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[!] 无法打开视频：{path}")
        return

    # 可选保存
    writer = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25
        w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(path.replace(".avi", "_labeled.avi"), fourcc, fps, (w,h))

    vote_q = deque(maxlen=SMOOTH_N)
    x,y,w,h = ROI
    roi_area = w * h

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 如果你的源视频倒置，可以打开下一行
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        # 取 ROI
        roi = frame[y:y+h, x:x+w].copy()
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        counts, masks = dominant_color_count(hsv)
        label_inst = decide_label(counts, roi_area)       # 当前帧的即时判断
        label = smooth_vote(vote_q, label_inst)           # 平滑后的最终显示

        # 叠加信息
        out = draw_overlay(frame, ROI, label, counts)

        # 可选展示掩码
        if SHOW_MASKS:
            m_red, m_blue, m_purple = masks
            cv2.imshow("mask_red", m_red)
            cv2.imshow("mask_blue", m_blue)
            cv2.imshow("mask_purple", m_purple)

        cv2.imshow(WINDOW_NAME, out)
        if writer is not None:
            writer.write(out)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:   # ESC 退出
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

# ========== 4) 执行 ==========
if __name__ == "__main__":
    for p in VIDEO_PATHS:
        print(f"==> Processing: {p}")
        process_video(p)
