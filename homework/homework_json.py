import cv2
import numpy as np
import json

# 保留你原来的 ColorRange
class ColorRange:
    def __init__(self, pic):
        self.hsv = cv2.cvtColor(pic, cv2.COLOR_BGR2HSV_FULL)
        self.ranges = {}

    def add_color_range(self, name, lower, upper):
        if name not in self.ranges:
            self.ranges[name] = []
        self.ranges[name].append((np.array(lower), np.array(upper)))

    def make_mask(self, name, KERNEL):
        if name not in self.ranges:
            return None
        masks = [cv2.inRange(self.hsv, lo, up) for lo, up in self.ranges[name]]
        mask = masks[0]
        for m in masks[1:]:
            mask = cv2.bitwise_or(mask, m)
        if KERNEL is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
        return mask


class BallDetector:
    def __init__(self, video_path, roi=None):
        self.cap = cv2.VideoCapture(video_path)
        self.roi = roi

    def _apply_roi(self, frame, mask):
        if not self.roi:
            return frame, mask
        x, y, w, h = self.roi
        return frame[y:y+h, x:x+w], mask[y:y+h, x:x+w]

    def run(self, ranges_dict, color_names,
            KERNEL=None, MIN_PIX=300, NO_BALL=0.01,
            rotate180=False, COLOR_TEXT=None):

        # 兜底 + 防守性转换（JSON 读出来是 list）
        if COLOR_TEXT is None:
            COLOR_TEXT = {"null": (200, 200, 200)}
        else:
            COLOR_TEXT = {k: tuple(v) for k, v in COLOR_TEXT.items()}

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if rotate180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            cr = ColorRange(frame)
            cr.ranges = ranges_dict

            dic = {}
            frame_roi = frame  # 给个默认，避免极端情况下未赋值
            for name in color_names:
                mask = cr.make_mask(name, KERNEL)
                if mask is None:
                    continue
                frame_roi, mask_roi = self._apply_roi(frame, mask)
                dic[name] = cv2.countNonZero(mask_roi)

            label = "null"
            if dic:
                roi_area = frame_roi.shape[0] * frame_roi.shape[1]
                check = {k: v for k, v in dic.items()
                         if v >= MIN_PIX and v/roi_area >= NO_BALL}
                if check:
                    label = max(dic, key=dic.get)

            if self.roi:
                x, y, w, h = self.roi
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.putText(frame, label, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        COLOR_TEXT.get(label, (0, 255, 0)), 2)
            cv2.imshow("Frame", frame)

            if cv2.waitKey(25) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()


def main():
    # 读取配置（加上编码更稳）
    with open("config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    ROI       = tuple(cfg["roi"])
    MIN_PIX   = cfg["min_pix"]
    NO_BALL   = cfg["no_ball"]
    ROTATE    = cfg["rotate180"]
    COLOR_TEXT = {k: tuple(v) for k, v in cfg["COLOR_TEXT"].items()}  # list→tuple

    # 设置颜色范围
    dummy = np.zeros((1, 1, 3), np.uint8)  # 占位 方便初始化
    cr_cfg = ColorRange(dummy)
    for name, (lower, upper) in cfg["color_ranges"].items():
        cr_cfg.add_color_range(name, lower, upper)

    # 运行检测（把 COLOR_TEXT 传进去）
    detector = BallDetector("res/output.avi", roi=ROI)
    KERNEL = np.ones((5, 5), np.uint8)
    detector.run(cr_cfg.ranges,
                 list(cfg["color_ranges"].keys()),
                 KERNEL, MIN_PIX, NO_BALL, ROTATE,
                 COLOR_TEXT=COLOR_TEXT)


if __name__ == "__main__":
    main()
