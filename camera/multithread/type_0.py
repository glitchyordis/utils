# change dir to this file path before running
# Adapted from https://github.com/rail-berkeley/hil-serl

import queue # threaded queue, not multiprocessing
import cv2
from pynput import keyboard
from utils import VideoCapture, ImageDisplayer

def on_press(key):
    global running
    if key == keyboard.Key.esc:
        running = False
        return False # required to stop listener

if __name__ == "__main__":
    """
    Utilize global running flag.

    thread_display
    - False: break while loop with 'q' key
    - True:
        - uses queue to show image
        - break while loop with 'esc' key, which sets `running` to False
    """
    thread_display = True

    running = True
    cap = cv2.VideoCapture(0)
    vid_cap = VideoCapture(cap)
    print("init camaera")

    if thread_display:
        img_queue = queue.Queue()
        img_displayer = ImageDisplayer(img_queue, "frame")
        img_displayer.start()

        listener = keyboard.Listener(on_press=on_press)
        listener.start()

    try:
        while running:
            frame = vid_cap.read()
            if frame is None:
                break
            if not thread_display:
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
        if thread_display:
            img_queue.put(None)
            img_displayer.join()
        cv2.destroyAllWindows()
        print("close camera")