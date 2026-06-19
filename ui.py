import cv2
from dataclasses import dataclass
from entities import Drawable, Skeleton
from video import Video

import math

@dataclass
class HUD(Drawable):
    skeleton: Skeleton
    status_text: str = ""
    detector_name: str = ""
    active_side: str = "Unknown"
    angles: dict = None

    def get_normalized_coords(self):
        if not self.skeleton:
            return {}, 1.0
        skel = self.skeleton
        l_hip = skel.landmarks.get(23)
        r_hip = skel.landmarks.get(24)
        l_sho = skel.landmarks.get(11)
        r_sho = skel.landmarks.get(12)
        
        scale = 1.0
        origin_x, origin_y = 0.0, 0.0
        
        if l_hip and r_hip and l_sho and r_sho:
            origin_x = (l_hip.x + r_hip.x) / 2.0
            origin_y = (l_hip.y + r_hip.y) / 2.0
            mid_sho_x = (l_sho.x + r_sho.x) / 2.0
            mid_sho_y = (l_sho.y + r_sho.y) / 2.0
            scale = math.hypot(mid_sho_x - origin_x, mid_sho_y - origin_y)
        else:
            xs = [lm.x for lm in skel.landmarks.values()]
            ys = [lm.y for lm in skel.landmarks.values()]
            if xs and ys:
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                origin_x, origin_y = min_x, min_y
                scale = max_y - min_y
                
        if scale < 1.0:
            scale = 1.0
            
        norm_coords = {}
        for idx, lm in skel.landmarks.items():
            nx = (lm.x - origin_x) / scale
            ny = (lm.y - origin_y) / scale
            norm_coords[idx] = (nx, ny, lm.visibility)
        return norm_coords, scale

    def draw(self, video: Video, frame):
        overlay = frame.copy()
        h, w = frame.shape[:2]
        ui_scale = h / 720.0
        
        box_w = int(330 * ui_scale)
        box_h = int(580 * ui_scale)
        box_x = w - box_w - int(20 * ui_scale)
        box_y = int(20 * ui_scale)

        # Draw professional semi-transparent panel
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (15, 15, 15), -1)
        # Draw border
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (255, 200, 0), max(1, int(2 * ui_scale)))

        if self.detector_name == "MediaPipe":
            lbox_w = int(330 * ui_scale)
            lbox_h = int(450 * ui_scale)
            lbox_x = int(20 * ui_scale)
            lbox_y = int(20 * ui_scale)
            cv2.rectangle(overlay, (lbox_x, lbox_y), (lbox_x + lbox_w, lbox_y + lbox_h), (15, 15, 15), -1)
            cv2.rectangle(overlay, (lbox_x, lbox_y), (lbox_x + lbox_w, lbox_y + lbox_h), (255, 200, 0), max(1, int(2 * ui_scale)))

        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

        font = cv2.FONT_HERSHEY_DUPLEX
        font_mono = cv2.FONT_HERSHEY_SIMPLEX
        font_scale_title = 0.6 * ui_scale
        font_scale = 0.45 * ui_scale
        font_scale_sm = 0.4 * ui_scale
        font_scale_xs = 0.35 * ui_scale
        thickness = max(1, int(1 * ui_scale))

        header_color = (255, 200, 0) # Azure/Cyan
        title_color = (255, 255, 255)
        text_color = (200, 200, 200)
        left_color = (100, 255, 100) # Bright Green
        right_color = (255, 100, 255) # Bright Magenta
        
        y_offset = box_y + int(35 * ui_scale)
        
        # Title
        cv2.putText(frame, "[ FREETHROW ANALYSIS ]", (box_x + int(35 * ui_scale), y_offset), font, font_scale_title, title_color, thickness + 1)
        y_offset += int(40 * ui_scale)

        # Handedness
        cv2.putText(frame, "> TARGET ACQUIRED <", (box_x + int(15 * ui_scale), y_offset), font, font_scale, header_color, thickness)
        y_offset += int(25 * ui_scale)
        hand_color = right_color if self.active_side == "Right" else left_color if self.active_side == "Left" else text_color
        cv2.putText(frame, f"HANDEDNESS: {self.active_side}", (box_x + int(15 * ui_scale), y_offset), font, font_scale, hand_color, thickness)
        y_offset += int(40 * ui_scale)

        # Angles
        cv2.putText(frame, "--- KINEMATIC ANGLES ---", (box_x + int(15 * ui_scale), y_offset), font, font_scale, header_color, thickness)
        y_offset += int(25 * ui_scale)
        
        cv2.putText(frame, "JOINT", (box_x + int(15 * ui_scale), y_offset), font, font_scale_xs, text_color, thickness)
        cv2.putText(frame, "LEFT", (box_x + int(120 * ui_scale), y_offset), font, font_scale_xs, text_color, thickness)
        cv2.putText(frame, "RIGHT", (box_x + int(210 * ui_scale), y_offset), font, font_scale_xs, text_color, thickness)
        y_offset += int(20 * ui_scale)

        if self.angles:
            def draw_angle(name, left_val, right_val, y):
                l_str = f"{int(left_val):3d} deg" if left_val is not None else " N/A "
                r_str = f"{int(right_val):3d} deg" if right_val is not None else " N/A "
                cv2.putText(frame, name, (box_x + int(15 * ui_scale), y), font_mono, font_scale_sm, title_color, thickness)
                cv2.putText(frame, l_str, (box_x + int(120 * ui_scale), y), font_mono, font_scale_sm, left_color, thickness)
                cv2.putText(frame, r_str, (box_x + int(210 * ui_scale), y), font_mono, font_scale_sm, right_color, thickness)

            draw_angle("Shoulder", self.angles.get("ls"), self.angles.get("rs"), y_offset)
            y_offset += int(25 * ui_scale)
            draw_angle("Elbow", self.angles.get("le"), self.angles.get("re"), y_offset)
            y_offset += int(25 * ui_scale)
            draw_angle("Knee", self.angles.get("lk"), self.angles.get("rk"), y_offset)
        
        y_offset += int(40 * ui_scale)

        # Coordinates
        cv2.putText(frame, "--- SPATIAL COORDINATES ---", (box_x + int(15 * ui_scale), y_offset), font, font_scale, header_color, thickness)
        y_offset += int(25 * ui_scale)
        
        cv2.putText(frame, "NODE", (box_x + int(15 * ui_scale), y_offset), font, font_scale_xs, text_color, thickness)
        cv2.putText(frame, "X", (box_x + int(120 * ui_scale), y_offset), font, font_scale_xs, text_color, thickness)
        cv2.putText(frame, "Y", (box_x + int(180 * ui_scale), y_offset), font, font_scale_xs, text_color, thickness)
        cv2.putText(frame, "CONF", (box_x + int(250 * ui_scale), y_offset), font, font_scale_xs, text_color, thickness)
        y_offset += int(25 * ui_scale)

        landmark_mapping = {
            11: "L.Sho", 12: "R.Sho", 13: "L.Elb", 14: "R.Elb",
            15: "L.Wri", 16: "R.Wri", 23: "L.Hip", 24: "R.Hip",
            25: "L.Kne", 26: "R.Kne", 27: "L.Ank", 28: "R.Ank"
        }
        
        norm_coords, _ = self.get_normalized_coords()
        
        for idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
            name = landmark_mapping[idx]
            is_left = idx % 2 != 0
            row_color = left_color if is_left else right_color
            
            cv2.putText(frame, name, (box_x + int(15 * ui_scale), y_offset), font_mono, font_scale_sm, row_color, thickness)
            
            if idx in norm_coords:
                nx, ny, conf = norm_coords[idx]
                x_str = f"{nx:+.2f}"
                y_str = f"{ny:+.2f}"
                conf_str = f"{conf*100:.0f}%"
                
                conf_color = (0, 255, 0) if conf > 0.5 else (0, 0, 255)
                
                cv2.putText(frame, x_str, (box_x + int(110 * ui_scale), y_offset), font_mono, font_scale_sm, title_color, thickness)
                cv2.putText(frame, y_str, (box_x + int(170 * ui_scale), y_offset), font_mono, font_scale_sm, title_color, thickness)
                cv2.putText(frame, conf_str, (box_x + int(250 * ui_scale), y_offset), font_mono, font_scale_sm, conf_color, thickness)
            else:
                cv2.putText(frame, " - ", (box_x + int(110 * ui_scale), y_offset), font_mono, font_scale_sm, text_color, thickness)
                cv2.putText(frame, " - ", (box_x + int(170 * ui_scale), y_offset), font_mono, font_scale_sm, text_color, thickness)
                cv2.putText(frame, "N/A", (box_x + int(250 * ui_scale), y_offset), font_mono, font_scale_sm, (0, 0, 255), thickness)
                
            y_offset += int(20 * ui_scale)

        if self.detector_name == "MediaPipe":
            lbox_x = int(20 * ui_scale)
            lbox_y = int(20 * ui_scale)
            ly_offset = lbox_y + int(35 * ui_scale)
            cv2.putText(frame, "[ MEDIAPIPE EXTRA ]", (lbox_x + int(45 * ui_scale), ly_offset), font, font_scale_title, title_color, thickness + 1)
            ly_offset += int(40 * ui_scale)

            cv2.putText(frame, "NODE", (lbox_x + int(15 * ui_scale), ly_offset), font, font_scale_xs, text_color, thickness)
            cv2.putText(frame, "X", (lbox_x + int(120 * ui_scale), ly_offset), font, font_scale_xs, text_color, thickness)
            cv2.putText(frame, "Y", (lbox_x + int(180 * ui_scale), ly_offset), font, font_scale_xs, text_color, thickness)
            cv2.putText(frame, "CONF", (lbox_x + int(250 * ui_scale), ly_offset), font, font_scale_xs, text_color, thickness)
            ly_offset += int(25 * ui_scale)

            extra_mapping = {
                0: "Nose", 2: "L.Eye", 5: "R.Eye", 7: "L.Ear", 8: "R.Ear",
                17: "L.Pinky", 18: "R.Pinky", 19: "L.Index", 20: "R.Index",
                21: "L.Thumb", 22: "R.Thumb", 29: "L.Heel", 30: "R.Heel",
                31: "L.FtIdx", 32: "R.FtIdx"
            }
            
            for e_idx in sorted(extra_mapping.keys()):
                name = extra_mapping[e_idx]
                is_left = (e_idx % 2 != 0) and (e_idx != 0)
                is_right = (e_idx % 2 == 0) and (e_idx != 0)
                row_col = left_color if is_left else right_color if is_right else text_color
                
                cv2.putText(frame, name, (lbox_x + int(15 * ui_scale), ly_offset), font_mono, font_scale_sm, row_col, thickness)
                
                if e_idx in norm_coords:
                    nx, ny, conf = norm_coords[e_idx]
                    cv2.putText(frame, f"{nx:+.2f}", (lbox_x + int(110 * ui_scale), ly_offset), font_mono, font_scale_sm, title_color, thickness)
                    cv2.putText(frame, f"{ny:+.2f}", (lbox_x + int(170 * ui_scale), ly_offset), font_mono, font_scale_sm, title_color, thickness)
                    conf_color = (0, 255, 0) if conf > 0.5 else (0, 0, 255)
                    cv2.putText(frame, f"{conf*100:.0f}%", (lbox_x + int(250 * ui_scale), ly_offset), font_mono, font_scale_sm, conf_color, thickness)
                else:
                    cv2.putText(frame, " - ", (lbox_x + int(110 * ui_scale), ly_offset), font_mono, font_scale_sm, text_color, thickness)
                    cv2.putText(frame, " - ", (lbox_x + int(170 * ui_scale), ly_offset), font_mono, font_scale_sm, text_color, thickness)
                    cv2.putText(frame, "N/A", (lbox_x + int(250 * ui_scale), ly_offset), font_mono, font_scale_sm, (0, 0, 255), thickness)
                    
                ly_offset += int(20 * ui_scale)


        # Big Status Text (Release / Prepare)
        if self.status_text:
            text = self.status_text
            (tw, th), _ = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_DUPLEX,
                1.5 * ui_scale,
                max(2, int(3 * ui_scale)),
            )
            center_x = int((w - tw) / 2)
            center_y = int((h + th) / 2)
            # Text shadow
            cv2.putText(frame, text, (center_x + 2, center_y + 2), cv2.FONT_HERSHEY_DUPLEX, 1.5 * ui_scale, (0, 0, 0), max(2, int(3 * ui_scale)))
            cv2.putText(frame, text, (center_x, center_y), cv2.FONT_HERSHEY_DUPLEX, 1.5 * ui_scale, (0, 0, 255), max(2, int(3 * ui_scale)))
            
            if self.detector_name:
                det_text = f"[{self.detector_name}]"
                (dtw, dth), _ = cv2.getTextSize(
                    det_text,
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.6 * ui_scale,
                    max(1, int(1.5 * ui_scale)),
                )
                cv2.putText(frame, det_text, (int((w - dtw) / 2) + 1, center_y + int(40 * ui_scale) + 1), cv2.FONT_HERSHEY_DUPLEX, 0.6 * ui_scale, (0, 0, 0), max(1, int(1.5 * ui_scale)))
                cv2.putText(frame, det_text, (int((w - dtw) / 2), center_y + int(40 * ui_scale)), cv2.FONT_HERSHEY_DUPLEX, 0.6 * ui_scale, (255, 200, 0), max(1, int(1.5 * ui_scale)))
