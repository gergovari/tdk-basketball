import cv2
import os
import threading
import queue
import time

class ThreadedVideoCapture:
    """Asynchronous video reader that decodes frames in a background thread.
    This prevents CPU decoding from blocking the GPU inference, significantly boosting FPS.
    """
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.q = queue.Queue(maxsize=128)
        self.stopped = False
        self.lock = threading.Lock()
        self.resize_to = None
        self.t = threading.Thread(target=self._reader, daemon=True)
        self.t.start()
        
    def set_resize(self, size):
        self.resize_to = size
        with self.q.mutex:
            self.q.queue.clear()

    def _reader(self):
        while not self.stopped:
            if not self.q.full():
                with self.lock:
                    ret, frame = self.cap.read()
                if not ret:
                    self.q.put((False, None))
                    self.stopped = True
                    break
                    
                if self.resize_to is not None:
                    frame = cv2.resize(frame, self.resize_to)
                    
                self.q.put((True, frame))
            else:
                time.sleep(0.005)

    def read(self):
        if self.stopped and self.q.empty():
            return False, None
        return self.q.get()

    def set(self, propId, value):
        with self.lock:
            res = self.cap.set(propId, value)
        if propId == cv2.CAP_PROP_POS_FRAMES:
            # Clear the queue when seeking so we don't get stale frames
            with self.q.mutex:
                self.q.queue.clear()
            self.stopped = False  # Reset stopped flag if we seek back
        return res

    def get(self, propId):
        with self.lock:
            return self.cap.get(propId)

    def isOpened(self):
        with self.lock:
            return self.cap.isOpened()

    def release(self):
        self.stopped = True
        with self.lock:
            if self.cap.isOpened():
                self.cap.release()

class Video:
    def __init__(self, input_video_path, output_video_path, scale=1.0):
        self.Scale = scale
        self.output_video_path = output_video_path
        self.cap = ThreadedVideoCapture(input_video_path)
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
        if hasattr(self.cap, 'set_resize'):
            self.cap.set_resize(self.size if self.scale != 1.0 else None)

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
                # Frame is already resized by the background thread if applicable!
                return frame
        raise StopIteration

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError("Video index out of range")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self.cap.read()
        if not ret:
            raise IndexError("Failed to read video frame")
        # For __getitem__, we might need to manually resize if the queue didn't catch it
        if self.scale != 1.0 and frame.shape[:2] != self.size[::-1]:
            frame = cv2.resize(frame, self.size)
        return frame
