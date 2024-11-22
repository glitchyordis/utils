# Adapted from https://github.com/rail-berkeley/hil-serl

import queue # threaded queue, not multiprocessing
import threading
import cv2
from pynput import keyboard

class VideoCapture:
    def __init__(self, cap, name=None):
        self.name = name
        self.q = queue.Queue()
        self.cap = cap
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = False
        self.enable = True
        self.t.start()

    def _reader(self):
        """
        get_nowait() method:
            Attempts to remove the oldest frame from the queue without blocking. 
            The get_nowait method retrieves and removes an item from the queue immediately. 
            If the queue is empty, it raises a queue.Empty exception.
        """
        while self.enable:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get(timeout=5)

    def close(self):
        self.enable = False
        self.t.join()
        self.cap.release()

class ImageDisplayer(threading.Thread):
    def __init__(self, queue, name):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True
        self.name = name

    def run(self):
        while True:
            img_array = self.queue.get()
            if img_array is None:  # signal to close
                break
            cv2.imshow(self.name, img_array)
            cv2.waitKey(1)

def on_press(running):
    def inner(key):
        if key == keyboard.Key.esc:
            running[0] = False
            return False  # Stop the listener
    return inner