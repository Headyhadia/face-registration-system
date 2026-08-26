"""
Face Registration System
-------------------------
Captures a student's face via webcam, verifies liveness (anti-spoofing)
before accepting the capture, and saves confirmed-live face images to
dataset/Student_<ID>/ for later use in a recognition system.

"""

import os
import re
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


def get_student_id() -> str:
    """
    Prompt for a Student ID and sanitize it into something safe to use
    as a folder name. Also warns (rather than silently overwrites) if
    that student already has a dataset folder.
    """
    while True:
        raw_id = input("Enter Student ID: ").strip()

        # Only allow letters, numbers, underscores, hyphens - keeps the
        # folder name filesystem-safe regardless of what the user types.
        student_id = re.sub(r"[^A-Za-z0-9_-]", "", raw_id)

        if not student_id:
            print("Student ID can't be empty (or contained only invalid characters). Try again.")
            continue

        target_folder = os.path.join(DATASET_DIR, f"Student_{student_id}")
        if os.path.exists(target_folder) and os.listdir(target_folder):
            choice = input(
                f"'{target_folder}' already has images. Overwrite? (y/n): "
            ).strip().lower()
            if choice != "y":
                continue  # ask for a fresh ID instead

        return student_id


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
    the registration gets live feedback: are they detected, are they being
    read as real or spoof, and how many images have been captured so far.
    Pure UI - no effect on the actual pipeline logic.
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
    Runs the live capture loop for one student. Returns True on a
    successful registration (enough live images saved), False if the
    user quit early or liveness was never established.
    """
    output_folder = os.path.join(DATASET_DIR, f"Student_{student_id}")
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not access the webcam. Check it's connected and not in use elsewhere.")
        return False

    window_name = "Face Registration - press q to cancel"
    # WINDOW_GUI_NORMAL (rather than the default) strips out OpenCV's extra
    # Qt/GTK toolbar (the zoom/save/properties row with the dropdown arrow)
    # that was causing the window to render blank until interacted with.
    # This also naturally sizes to the actual webcam resolution, restoring
    # the original correct size instead of the fixed 960x720.
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

    # Pump the GUI event loop every ~50ms so the OS doesn't consider the
    # window unresponsive while we wait, without trying to render live
    # video during this phase.
    while not model_ready.is_set():
        cv2.waitKey(50)

    frame_count = 0
    consecutive_live = 0
    saved_count = 0
    last_result = None  # reused between the (less frequent) liveness checks so the overlay doesn't flicker
    ever_saw_spoof = False
    locked_box = None  # the position of the face we've committed to tracking, once liveness starts building

    print("\nLoading models in the background - camera preview is already live.")
    print("Look at the camera. Hold still and well-lit for best results.")
    print("Registration begins automatically once liveness is confirmed.\n")

    try:
        while saved_count < IMAGES_TO_CAPTURE:
            ok, frame = cap.read()
            if not ok:
                print("ERROR: Lost connection to the webcam.")
                break

            frame_count += 1

            if not model_ready.is_set():
                # Model's still loading in the background - show live video
                # with a status overlay, but skip detection entirely so we
                # don't touch DeepFace until the background thread is done
                # warming it up (avoids loading it twice/racing on it).
                cv2.putText(
                    frame, "Loading models, please wait...", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2,
                )
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\nCancelled by user.")
                    break
                continue

            # Anti-spoofing inference is the expensive step, so we don't run it
            # on every single frame - just often enough to react quickly while
            # keeping the preview smooth.
            if frame_count % LIVENESS_CHECK_EVERY_N_FRAMES == 0:
                last_result = analyze_frame(frame, locked_box)

                if last_result is None or not last_result["same_identity"] and locked_box is not None:
                    # Either nobody's there, or the tracked person appears to
                    # have been replaced by someone else in frame - don't let
                    # a bystander's face silently continue the count.
                    consecutive_live = 0
                    locked_box = None
                elif last_result["is_real"]:
                    consecutive_live += 1
                    locked_box = last_result["box"]  # keep tracking this exact face going forward
                else:
                    consecutive_live = 0
                    locked_box = last_result["box"]  # still lock on - person is there, just not confirmed live yet
                    ever_saw_spoof = True

            display_frame = frame.copy()
            draw_overlay(display_frame, last_result, consecutive_live, saved_count)
            cv2.imshow(window_name, display_frame)

            # Only start saving once seen enough consecutive live reads -
            # this is what will makes a single lucky misclassification insufficient
            # to fool the system, and what makes a held-up photo/screen (which
            # reads consistently as spoof, not randomly) get reliably rejected.
            if (
                consecutive_live >= REQUIRED_CONSECUTIVE_LIVE
                and frame_count % CAPTURE_EVERY_N_FRAMES == 0
            ):
                x, y, w, h = last_result["box"]
                # Small margin around the tight detection box - standard
                # practice for face datasets, since a bit of context around
                # the face (not just eyes-nose-mouth) tends to help whatever
                # recognition model consumes this dataset later. Clamped to
                # the frame edges so it never reads outside the image bounds.
                margin = int(0.2 * max(w, h))
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(frame.shape[1], x + w + margin)
                y2 = min(frame.shape[0], y + h + margin)
                face_crop = frame[y1:y2, x1:x2]

                image_path = os.path.join(output_folder, f"image_{saved_count + 1}.jpg")
                cv2.imwrite(image_path, face_crop)
                saved_count += 1
                print(f"Captured image {saved_count}/{IMAGES_TO_CAPTURE}")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nCancelled by user.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    if saved_count >= IMAGES_TO_CAPTURE:
        print("Face registration successful.")
        print(f"{saved_count} images saved to '{output_folder}'.")
        return True

    # Distinguish *why* it failed - a spoof attempt vs. a plain early quit -
    if ever_saw_spoof and consecutive_live == 0:
        print("\nWARNING: Live person not detected (possible spoof attempt). Registration cancelled.")
    else:
        print("\nRegistration cancelled before enough images were captured.")
    return False


def main():
    os.makedirs(DATASET_DIR, exist_ok=True)
    student_id = get_student_id()
    run_registration(student_id)


if __name__ == "__main__":
    main()