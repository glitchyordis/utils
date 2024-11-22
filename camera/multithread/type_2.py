# change dir to this file path before running
# Adapted from https://github.com/rail-berkeley/hil-serl

import queue # threaded queue, not multiprocessing
import cv2
from utils import VideoCapture, ImageDisplayer
from pynput import keyboard

class VideoApp:
    def __init__(self):
        self.thread_display = False
        self.running = True

    def on_press(self, key):
        if key == keyboard.Key.esc:
            self.running = False
            return False  # Stop the listener

    def run(self):
        cap = cv2.VideoCapture(0)
        vid_cap = VideoCapture(cap)
        print("init camera")

        if self.thread_display:
            img_queue = queue.Queue()
            img_displayer = ImageDisplayer(img_queue, "frame")
            img_displayer.start()

        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()

        try:
            while self.running:
                frame = vid_cap.read()
                if frame is None:
                    break
                if not self.thread_display:
                    cv2.imshow("frame", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("break")
                        break
                else:
                    img_queue.put(frame)
        except Exception as e:
            print(f"exception: {e}")
        finally:
            vid_cap.close()
            if self.thread_display:
                img_queue.put(None)  # signal the ImageDisplayer thread to exit
                img_displayer.join()  # wait for the ImageDisplayer thread to finish
            cv2.destroyAllWindows()
            print("close camera")

if __name__ == "__main__":
    """
    This version uses a class to encapsulate the video app. with 
    running flag as an attribute.
    """
    app = VideoApp()
    app.run()