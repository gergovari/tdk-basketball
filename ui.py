import cv2
from dataclasses import dataclass
from entities import Drawable, Skeleton
from video import Video

@dataclass
class HUD(Drawable):
    skeleton: Skeleton
    released: bool = False
    detector_name: str = ""

    def draw(self, video: Video, frame):
        overlay = frame.copy()
        h, w = frame.shape[:2]
        ui_scale = h / 720.0
        box_w, box_h = int(250 * ui_scale), int(150 * ui_scale)
        box_x, box_y = w - box_w - int(20 * ui_scale), int(20 * ui_scale)

        cv2.rectangle(
            overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1
        )
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5 * ui_scale
        thickness = max(1, int(1 * ui_scale))

        left_color = (0, 165, 255)  # Orange
        right_color = (255, 0, 0)  # Blue
        text_color = (255, 255, 255)

        y_offset = box_y + int(25 * ui_scale)

        active_side = "Unknown"
        if self.skeleton:
            if 15 in self.skeleton.landmarks and 16 in self.skeleton.landmarks:
                if self.skeleton.landmarks[16].y < self.skeleton.landmarks[15].y:
                    active_side = "Right"
                else:
                    active_side = "Left"

        cv2.putText(
            frame,
            f"Hand: {active_side}",
            (box_x + int(10 * ui_scale), y_offset),
            font,
            font_scale,
            text_color,
            thickness,
        )
        y_offset += int(25 * ui_scale)

        if self.skeleton:

            def draw_angle(name, left_val, right_val, y):
                l_str = f"{int(left_val)}" if left_val is not None else "N/A"
                r_str = f"{int(right_val)}" if right_val is not None else "N/A"
                cv2.putText(
                    frame,
                    f"{name}:",
                    (box_x + int(10 * ui_scale), y),
                    font,
                    font_scale,
                    text_color,
                    thickness,
                )
                cv2.putText(
                    frame,
                    l_str,
                    (box_x + int(120 * ui_scale), y),
                    font,
                    font_scale,
                    left_color,
                    thickness,
                )
                cv2.putText(
                    frame,
                    r_str,
                    (box_x + int(180 * ui_scale), y),
                    font,
                    font_scale,
                    right_color,
                    thickness,
                )

            draw_angle(
                "Shoulder",
                self.skeleton.left_shoulder_angle,
                self.skeleton.right_shoulder_angle,
                y_offset,
            )
            y_offset += int(25 * ui_scale)
            draw_angle(
                "Elbow",
                self.skeleton.left_elbow_angle,
                self.skeleton.right_elbow_angle,
                y_offset,
            )
            y_offset += int(25 * ui_scale)
            draw_angle(
                "Knee",
                self.skeleton.left_knee_angle,
                self.skeleton.right_knee_angle,
                y_offset,
            )

        if self.released:
            text = "RELEASED"
            (tw, th), _ = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5 * ui_scale,
                max(2, int(3 * ui_scale)),
            )
            center_x = int((w - tw) / 2)
            center_y = int((h + th) / 2)
            cv2.putText(
                frame,
                text,
                (center_x, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5 * ui_scale,
                (0, 0, 255),
                max(2, int(3 * ui_scale)),
            )
            if self.detector_name:
                det_text = f"({self.detector_name})"
                (dtw, dth), _ = cv2.getTextSize(
                    det_text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7 * ui_scale,
                    max(1, int(1.5 * ui_scale)),
                )
                cv2.putText(
                    frame,
                    det_text,
                    (int((w - dtw) / 2), center_y + int(40 * ui_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7 * ui_scale,
                    (0, 255, 0),
                    max(1, int(1.5 * ui_scale)),
                )
