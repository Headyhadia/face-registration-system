"""
Face Registration System
-------------------------
Captures a student's face via webcam, verifies liveness (anti-spoofing)
before accepting the capture, and saves confirmed-live face images to
dataset/Student_<ID>/ for later use in a recognition system.

"""

import os
import re
import time
import threading
import numpy as np

# log levels are set to 3 to suppress TensorFlow and OpenCV debug output, which is
# very verbose and not useful for the user.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")

import cv2
from deepface import DeepFace

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

DATASET_DIR = "dataset"                # where per-student folders get created
DETECTOR_BACKEND = "opencv"            # face detector DeepFace uses internally
IMAGES_TO_CAPTURE = 8                  # how many confirmed-live face crops to save
REQUIRED_CONSECUTIVE_LIVE = 6          # liveness checks in a row before we start saving images
LIVENESS_CHECK_EVERY_N_FRAMES = 5      # run the model every Nth frame, not every frame
CAPTURE_EVERY_N_FRAMES = 8             # spacing between saved images, for slight pose variety
ANTISPOOF_CONFIDENCE_THRESHOLD = 0.70  # DeepFace's own score; frames below this count as "spoof"
MAX_REGISTRATION_SECONDS = 25          # overall time limit per registration attempt
IDENTITY_LOST_GRACE_SECONDS = 2.5      # how long we wait after losing the tracked face before cancelling
def get_student_id() -> str:
    """
    Prompt for a Student ID, validated to be filesystem-safe.
    If invalid characters entered, we reject and re-prompt so the
    person knows what happened.
    """
    valid_pattern = re.compile(r"^[A-Za-z0-9_-]+$")

    while True:
        raw_id = input("Enter Student ID (letters, numbers, _ and - only): ").strip()

        if not raw_id:
            print("Student ID can't be empty. Try again.")
            continue

        if not valid_pattern.match(raw_id):
            print(f"'{raw_id}' contains characters that aren't allowed (only letters, numbers, _, - are permitted). Try again.")
            continue

        target_folder = os.path.join(DATASET_DIR, f"Student_{raw_id}")
        if os.path.exists(target_folder) and os.listdir(target_folder):
            choice = input(
                f"'{target_folder}' already has images. Overwrite? (y/n): "
            ).strip().lower()
            if choice != "y":
                continue

        return raw_id
    
def box_iou(box_a, box_b):
    """
    Intersection-over-Union between two (x, y, w, h) boxes: 0 means no
    overlap at all, 1 means identical boxes. Used to answer "is this the
    same face as last time, or did a different person just appear."
    """
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax + aw, bx + bw)
    inter_y2 = min(ay + ah, by + bh)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union_area = (aw * ah) + (bw * bh) - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def select_tracked_face(face_objs, locked_box):
    """
    Chooses which detected face to treat as "the person being registered."

    First registration attempt in a session (locked_box is None): pick the
    largest face, the person sitting down to register is
    almost always the biggest/closest face.

    Once we're already tracking someone (locked_box is set): stick with
    whichever detected face overlaps most with where that person was last
    seen, rather than switching to whoever is now biggest. If a bystander
    leans in and becomes the largest face, this keeps the lock on the
    original person instead of silently swapping identities mid-capture.

    Returns (chosen_face, is_same_identity) - is_same_identity is False
    when either there was no previous lock, or the best-overlapping face
    doesn't overlap enough to plausibly be the same person (meaning the
    original person likely stepped out of frame).
    """
    if locked_box is None:
        face = max(face_objs, key=lambda f: f["facial_area"]["w"] * f["facial_area"]["h"])
        return face, False

    def box_of(f):
        a = f["facial_area"]
        return (a["x"], a["y"], a["w"], a["h"])

    best_face = max(face_objs, key=lambda f: box_iou(box_of(f), locked_box))
    best_iou = box_iou(box_of(best_face), locked_box)

    # Below this, treat it as "not the same person anymore" rather than
    # trusting a weak overlap - a bystander's face passing near the locked
    # position shouldn't be mistaken for the original person.
    return best_face, best_iou >= 0.3


def analyze_frame(frame, locked_box):
    """
    Runs face detection + anti-spoofing on a single frame, and resolves
    which detected face is the tracked person via select_tracked_face.
    Returns a dict describing what was found, or None if nothing usable
    was found (no face, or DeepFace couldn't process it).

    Wrapping DeepFace's call in try/except raises an
    exception (rather than returning an empty list) when no face is found
    in the frame, which is an *expected*, normal event in a live webcam
    loop (person steps out of frame, blinks, moves too fast) - it
    should not crash the program.
    """
    try:
        face_objs = DeepFace.extract_faces(
            img_path=frame,
            detector_backend=DETECTOR_BACKEND,
            anti_spoofing=True,
            enforce_detection=True,
        )
    except ValueError:
        # DeepFace raises this when no face is detected in the frame at all.
        return None

    if not face_objs:
        return None

    face, same_identity = select_tracked_face(face_objs, locked_box)

    area = face["facial_area"]
    is_real = face.get("is_real", False)
    antispoof_score = face.get("antispoof_score", 0.0)

    return {
        "box": (area["x"], area["y"], area["w"], area["h"]),
        "is_real": bool(is_real) and antispoof_score >= ANTISPOOF_CONFIDENCE_THRESHOLD,
        "score": antispoof_score,
        "same_identity": same_identity,
    }


def draw_overlay(frame, result, consecutive_live, saved_count):
    """
    Draws the bounding box + status text on the frame so the person doing
    the registration gets live feedback. Pure UI - never touches the saved
    image data (that's handled separately with a clean copy of the frame).
    """
    if result is None:
        cv2.putText(
            frame, "No face detected", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
        )
    else:
        x, y, w, h = result["box"]
        if result["is_real"]:
            color = (0, 200, 0)
            if consecutive_live >= REQUIRED_CONSECUTIVE_LIVE:
                # Once the streak requirement is already met, showing the
                # raw (still-climbing) count looks like a bug - switch to a
                # clear "capturing" status instead.
                label = f"LIVE - capturing ({saved_count}/{IMAGES_TO_CAPTURE})"
            else:
                label = f"REAL ({result['score']:.2f}) - {consecutive_live}/{REQUIRED_CONSECUTIVE_LIVE}"
        else:
            color = (0, 0, 255)
            label = f"SPOOF SUSPECTED ({result['score']:.2f})"

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            frame, label, (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )

    cv2.putText(
        frame, f"Captured: {saved_count}/{IMAGES_TO_CAPTURE}", (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
    )
    cv2.putText(
        frame, "Press 'q' to cancel", (20, frame.shape[0] - 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
    )

def run_registration(student_id: str) -> bool:
    """
    Runs the live capture loop for one student.

    Captured face crops are held in memory and only written to disk once
    the entire registration succeeds - a cancelled, timed-out, or
    identity-interrupted attempt leaves no folder and no partial images
    behind, and never mixes two different people's faces into one
    student's dataset.
    """
    output_folder = os.path.join(DATASET_DIR, f"Student_{student_id}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not access the webcam. Check it's connected and not in use elsewhere.")
        return False

    window_name = "Face Registration - press q to cancel"
    cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)

    loading_screen = np.zeros((480, 640, 3), dtype="uint8")
    cv2.putText(
        loading_screen, "Loading models, please wait...", (30, 240),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2,
    )
    cv2.imshow(window_name, loading_screen)
    cv2.waitKey(1)

    ok, warm_frame = cap.read()
    model_ready = threading.Event()

    def warm_up():
        if ok:
            analyze_frame(warm_frame, locked_box=None)
        model_ready.set()

    threading.Thread(target=warm_up, daemon=True).start()
    while not model_ready.is_set():
        cv2.waitKey(50)

    frame_count = 0
    consecutive_live = 0
    last_result = None
    ever_saw_spoof = False
    locked_box = None
    had_lock = False           # have we ever established a tracked identity this session
    lost_since = None          # timestamp when tracking was first lost, or None if currently tracked
    captured_crops = []        # buffered in memory - nothing written to disk until success
    start_time = time.time()
    outcome = None              # 'success' | 'user_quit' | 'timeout_spoof' | 'timeout_generic' | 'identity_lost'

    print("\nLook at the camera. Hold still and well-lit for best results.")
    print("Registration begins automatically once liveness is confirmed.\n")

    try:
        while len(captured_crops) < IMAGES_TO_CAPTURE:
            ok, frame = cap.read()
            if not ok:
                print("ERROR: Lost connection to the webcam.")
                outcome = "user_quit"
                break

            frame_count += 1

            if time.time() - start_time > MAX_REGISTRATION_SECONDS:
                outcome = "timeout_spoof" if ever_saw_spoof else "timeout_generic"
                break

            if frame_count % LIVENESS_CHECK_EVERY_N_FRAMES == 0:
                last_result = analyze_frame(frame, locked_box)

                lost_this_check = (
                    last_result is None
                    or (locked_box is not None and not last_result["same_identity"])
                )

                if lost_this_check:
                    consecutive_live = 0
                    if lost_since is None:
                        lost_since = time.time()
                else:
                    lost_since = None  # tracking recovered - grace period cancelled
                    locked_box = last_result["box"]
                    had_lock = True
                    if last_result["is_real"]:
                        consecutive_live += 1
                    else:
                        consecutive_live = 0
                        ever_saw_spoof = True

                if had_lock and lost_since is not None and (time.time() - lost_since) > IDENTITY_LOST_GRACE_SECONDS:
                    # Gave the person a real window to step back into frame
                    # after a momentary loss - if they haven't stepped by now,
                    # treat it as genuinely gone rather than a brief glitch.
                    outcome = "identity_lost"
                    break

            display_frame = frame.copy()
            draw_overlay(display_frame, last_result, consecutive_live, len(captured_crops))
            if lost_since is not None:
                remaining = max(0, IDENTITY_LOST_GRACE_SECONDS - (time.time() - lost_since))
                cv2.putText(
                    display_frame, f"Face lost - return within {remaining:.1f}s",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2,
                )
            cv2.imshow(window_name, display_frame)

            if (
                consecutive_live >= REQUIRED_CONSECUTIVE_LIVE
                and frame_count % CAPTURE_EVERY_N_FRAMES == 0
            ):
                x, y, w, h = last_result["box"]
                margin = int(0.2 * max(w, h))
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(frame.shape[1], x + w + margin)
                y2 = min(frame.shape[0], y + h + margin)
                captured_crops.append(frame[y1:y2, x1:x2].copy())
                print(f"Captured image {len(captured_crops)}/{IMAGES_TO_CAPTURE}")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                outcome = "user_quit"
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    if outcome is None and len(captured_crops) >= IMAGES_TO_CAPTURE:
        outcome = "success"

    if outcome == "success":
        os.makedirs(output_folder, exist_ok=True)
        for i, crop in enumerate(captured_crops, start=1):
            cv2.imwrite(os.path.join(output_folder, f"image_{i}.jpg"), crop)
        print("\nFace registration successful.")
        return True

    messages = {
        "user_quit": "\nRegistration cancelled by user.",
        "timeout_spoof": "\nWARNING: Live person not detected. Face registration cancelled.",
        "timeout_generic": "\nNo live face confirmed within the time limit. Registration cancelled.",
        "identity_lost": "\nThe tracked face changed or left the frame. Registration cancelled for safety - please try again.",
    }
    print(messages.get(outcome, "\nRegistration cancelled before enough images were captured."))
    return False

def main():
    student_id = get_student_id()
    run_registration(student_id)


if __name__ == "__main__":
    main()