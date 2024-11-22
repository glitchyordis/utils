# change dir to this file path before running
# Adapted from https://github.com/rail-berkeley/hil-serl

import queue # threaded queue, not multiprocessing
import cv2
from utils import VideoCapture, ImageDisplayer, on_press
from pynput import keyboard

if __name__ == "__main__":
    """
    This version uses a flag variable to control running loop
    """
    running = [True]  # Use a list to allow modification within the closure
    thread_display = True

    cap = cv2.VideoCapture(0)
    vid_cap = VideoCapture(cap)
    print("init camera")

    if thread_display:
        img_queue = queue.Queue()
        img_displayer = ImageDisplayer(img_queue, "frame")
        img_displayer.start()

    listener = keyboard.Listener(on_press=on_press(running))
    listener.start()

    try:
        while running[0]:
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
            img_queue.put(None)  # signal the ImageDisplayer thread to exit
            img_displayer.join()  # wait for the ImageDisplayer thread to finish
        cv2.destroyAllWindows()
        print("close camera")