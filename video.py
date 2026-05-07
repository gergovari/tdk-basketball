import cv2
import os

class Video:
    def __init__(self, input_video_path, output_video_path, scale=1.0):
        self.Scale = scale
        self.output_video_path = output_video_path
        self.cap = cv2.VideoCapture(input_video_path)
        self.out = cv2.VideoWriter(
            self.output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.size
        )
        self.frames_written = 0

    @property
    def size(self):
        width = int(int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * self.scale)
        height = int(int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * self.scale)
        return (width, height)

    @property
    def scale(self):
        return self.Scale

    @scale.setter
    def scale(self, new_scale):
        if hasattr(self, "out"):
            self.out.release()

        temp_old_video = self.output_video_path + ".old.mp4"
        has_valid_old_video = False

        if os.path.exists(self.output_video_path):
            if self.frames_written > 0:
                os.rename(self.output_video_path, temp_old_video)
                has_valid_old_video = True
            else:
                os.remove(self.output_video_path)

        self.Scale = new_scale

        self.out = cv2.VideoWriter(
            self.output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.size
        )

        if has_valid_old_video and os.path.exists(temp_old_video):
            old_cap = cv2.VideoCapture(temp_old_video)

            while old_cap.isOpened():
                ret, frame = old_cap.read()
                if not ret:
                    break

                resized_frame = cv2.resize(frame, self.size)
                self.out.write(resized_frame)

            old_cap.release()
            os.remove(temp_old_video)

    def __len__(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def fps(self):
        return int(self.cap.get(cv2.CAP_PROP_FPS))

    def write(self, frame):
        self.out.write(frame)
        self.frames_written += 1

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
        if hasattr(self, "out"):
            self.out.release()

    def __del__(self):
        self.release()

    def __iter__(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return self

    def __next__(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame if self.scale == 1.0 else cv2.resize(frame, self.size)
        raise StopIteration

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError("Video index out of range")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self.cap.read()
        if not ret:
            raise IndexError("Failed to read video frame")
        return frame if self.scale == 1.0 else cv2.resize(frame, self.size)
