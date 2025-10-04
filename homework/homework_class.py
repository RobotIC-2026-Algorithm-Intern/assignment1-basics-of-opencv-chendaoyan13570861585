import cv2
import numpy as np

class ColorRange:
    full = True  # 0-255说是（HSV_FULL 模式下 H 也是 0-255）

    def __init__(self, pic):
        # 保存原图 & 转 HSV（FULL）
        self.pic = pic
        self.hsv = cv2.cvtColor(pic, cv2.COLOR_BGR2HSV_FULL)
        self.ranges = {}  # 简单点：{name: [(lower, upper), ...]}

    def add_color_range(self, name, lower, upper):
        if name not in self.ranges:
            self.ranges[name] = []
        self.ranges[name].append((np.array(lower), np.array(upper)))
        return self.ranges  # 方便外面拿到当前所有区间

    def make_mask(self, name, KERNEL):
        if name not in self.ranges:
            return None
        masks = [cv2.inRange(self.hsv, lower, upper) for lower, upper in self.ranges[name]]
        full_mask = masks[0]
        for m in masks[1:]:
            full_mask = cv2.bitwise_or(full_mask, m)
        if KERNEL is not None:
            full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, KERNEL)
        return full_mask


class BallDetector:
    """视频检测逻辑"""

    def __init__(self, video_path, roi=None):
        self.cap = cv2.VideoCapture(video_path)
        self.roi = roi
        if not self.cap.isOpened():
            print(f"[错误] 无法打开视频：{video_path}")

    def _apply_roi(self, frame, mask):
        """裁剪 ROI"""
        if not self.roi:
            return frame, mask
        x, y, w, h = self.roi
        return frame[y:y+h, x:x+w], mask[y:y+h, x:x+w]

    def run(self, ranges_dict, color_names,
            KERNEL=None, MIN_PIX=300, NO_BALL=0.01,
            COLOR_TEXT=None, rotate180=False):

        if COLOR_TEXT is None:
            COLOR_TEXT = {"null": (200, 200, 200)}

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if rotate180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            cr = ColorRange(frame)
            cr.ranges = ranges_dict  # 注入颜色范围

            dic = {}
            for name in color_names:
                mask = cr.make_mask(name, KERNEL)
                if mask is None:
                    continue
                frame_roi, mask_roi = self._apply_roi(frame, mask)
                dic[name] = cv2.countNonZero(mask_roi)

            # 默认无球
            label = "null"
            if dic:
                roi_area = frame_roi.shape[0] * frame_roi.shape[1]  #长x宽
                # 筛选符合阈值的颜色
                check = {k:v for k,v in dic.items()
                         if v >= MIN_PIX and v/roi_area >= NO_BALL}
                if check:
                    label = max(dic, key=dic.get)

            # 绘制结果
            if self.roi:
                x, y, w, h = self.roi
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, label, (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        COLOR_TEXT.get(label, (0,255,0)), 2)

            cv2.imshow("Frame", frame)
            if cv2.waitKey(25) & 0xFF == 27:  # ESC退出
                break

        self.cap.release()
        cv2.destroyAllWindows()


def main():
    # 参数
    ROI = (200, 170, 200, 200)
    VIDEO_PATH = "res/output.avi"
    KERNEL = np.ones((5, 5), np.uint8)
    MIN_PIX, NO_BALL = 300, 0.01
    COLOR_TEXT = {"null": (200,200,200),
                  "red": (0,0,255),
                  "blue": (255,0,0),
                  "purple": (255,0,255)}

    # 定义颜色范围
    dummy = np.zeros((10,10,3), dtype=np.uint8)
    cr_cfg = ColorRange(dummy)
    cr_cfg.add_color_range("red",    (220,30,150), (250,255,255))
    cr_cfg.add_color_range("blue",   (142,80,80),  (184,255,255))
    cr_cfg.add_color_range("purple", (184,60,60),  (220,255,255))

    # 运行
    detector = BallDetector(VIDEO_PATH, roi=ROI)
    detector.run(cr_cfg.ranges, ["red","blue","purple"],
                 KERNEL=KERNEL, MIN_PIX=MIN_PIX, NO_BALL=NO_BALL,
                 COLOR_TEXT=COLOR_TEXT, rotate180=True)


if __name__ == "__main__":
    main()
