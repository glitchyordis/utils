import ctypes
import sys
import time

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import win32api
import win32con
import win32gui
from win32api import GetMonitorInfo

"""
This script moves a window with a given title (here, "MuJoCo") that was already launched/ with a wait time,  
to a specified monitor on Windows. 

Can take a target monitor index as input or allow the user to choose one interactively.
"""
target_display_idx = 1  # set to None for interactive selection
win_name = "MuJoCo"

# Script starts
# make coordinates DPI-aware so monitor rects match pixels
ctypes.windll.user32.SetProcessDPIAware()

# Wait briefly to ensure the MuJoCo viewer window exists
time.sleep(0.5)

hwnd = win32gui.FindWindow(None, win_name)
if not hwnd:
    # fallback: enumerate windows and look for a substring
    def _cb(h, acc):
        title = win32gui.GetWindowText(h)
        if win_name in title:
            acc.append(h)

    acc = []
    win32gui.EnumWindows(_cb, acc)
    hwnd = acc[0] if acc else None

if not hwnd:
    print(f"{win_name = } not found; cannot move.")
    sys.exit(1)


def choose_monitor_interactively():
    mons = list(win32api.EnumDisplayMonitors())
    # sort left->right for consistent ordering
    mons_sorted = sorted(mons, key=lambda m: m[2][0])
    rects = []
    devices = []
    for idx, (hmon, hdc, rect) in enumerate(mons_sorted, start=1):
        left, top, right, bottom = rect
        rects.append((left, top, right, bottom))
        info = GetMonitorInfo(hmon)
        devices.append(info.get("Device", f"MON{idx}"))
    # compute bounding box for plotting
    xs = [r[0] for r in rects] + [r[2] for r in rects]
    ys = [r[1] for r in rects] + [r[3] for r in rects]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, r in enumerate(rects, start=1):
        left, top, right, bottom = r
        w, h = right - left, bottom - top
        # matplotlib y goes up; Windows y goes down, so invert later
        rect_patch = patches.Rectangle(
            (left, top),
            w,
            h,
            linewidth=2,
            edgecolor="C{}".format(i % 10),
            facecolor="none",
        )
        ax.add_patch(rect_patch)
        ax.text(
            left + 10,
            top + 20,
            f"{i}: {devices[i - 1]} ({w}x{h})",
            color="C0",
            fontsize=9,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
    ax.set_xlim(min_x - 50, max_x + 50)
    ax.set_ylim(max_y + 50, min_y - 50)  # invert y axis by swapping limits
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f'Monitors (left->right). Enter index to move the "{win_name}" window to.'
    )
    ax.set_xlabel("pixels (x)")
    ax.set_ylabel("pixels (y)")
    plt.show(block=False)

    # ask user
    while True:
        s = input(f"Choose monitor index [1..{len(rects)}], or 'q' to cancel: ").strip()
        if s.lower() == "q":
            return None
        try:
            idx = int(s)
            if 1 <= idx <= len(rects):
                return idx, mons_sorted
        except Exception:
            pass
        print("Invalid choice.")


mons_sorted = sorted(list(win32api.EnumDisplayMonitors()), key=lambda m: m[2][0])

if target_display_idx is not None:
    # target_display_idx is 1-based in this script
    if 1 <= target_display_idx <= len(mons_sorted):
        target_index = target_display_idx
        print(f"Using target_display_idx = {target_display_idx}")
        # keep mons_sorted for later use
    else:
        print(
            f"target_display_idx {target_display_idx} out of range (1..{len(mons_sorted)}). Falling back to interactive selection."
        )
        choice = choose_monitor_interactively()
        if choice is None:
            print("User cancelled monitor selection.")
            target_index = None
        else:
            target_index, mons_sorted = choice
else:
    choice = choose_monitor_interactively()
    if choice is None:
        print("User cancelled monitor selection.")
        target_index = None
    else:
        target_index, mons_sorted = choice

if target_index is None:
    print("No target monitor selected; exiting.")
else:
    # get rect for selected monitor (1-based index)
    _, _, rect = mons_sorted[target_index - 1]
    left, top, right, bottom = rect
    mon_w, mon_h = right - left, bottom - top
    x, y, w, h = left, top, mon_w, mon_h

    if hwnd:
        win32gui.SetWindowPos(
            hwnd,
            win32con.HWND_TOP,
            x,
            y,
            w,
            h,
            win32con.SWP_NOZORDER | win32con.SWP_SHOWWINDOW | win32con.SWP_FRAMECHANGED,
        )
        # optionally maximize into monitor work area
        win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
        print(f"Moved window to monitor {target_index} at {x},{y} size {w}x{h}")
    else:
        print(f"{win_name} window not found; cannot move.")
