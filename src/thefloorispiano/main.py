import argparse
import json
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pygame
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = Path(__file__).resolve().parents[2] / "pose_landmarker.task"
LAYOUT_PATH = Path(__file__).resolve().parents[2] / "floor_layout.json"

DEFAULT_ZONE = [(0.20, 0.78), (0.80, 0.78), (0.96, 0.98), (0.04, 0.98)]
DEFAULT_NOTE = "C4"
DRAG_RADIUS = 15
FOOT_IDS = (27, 29, 31, 28, 30, 32)

ENTRY_DWELL = 0.18   # seconds continuously inside before triggering
EXIT_DWELL = 0.12    # seconds continuously outside before stopping

CONNECTIONS = (
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
    (24, 26), (26, 28), (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22), (27, 29), (29, 31),
    (28, 30), (30, 32),
)

NOTE_FREQUENCIES = {
    "C4": 261.63,
    "C#4": 277.18,
    "D4": 293.66,
    "D#4": 311.13,
    "E4": 329.63,
    "F4": 349.23,
    "F#4": 369.99,
    "G4": 392.00,
    "G#4": 415.30,
    "A4": 440.00,
    "A#4": 466.16,
    "B4": 493.88,
    "C5": 523.25,
}


def load_zones():
    if LAYOUT_PATH.exists():
        try:
            data = json.loads(LAYOUT_PATH.read_text())

            if isinstance(data.get("polygons"), list):
                zones = [
                    {
                        "vertices": [
                            (float(x), float(y))
                            for x, y in item["vertices"]
                        ],
                        "note": str(item.get("note", DEFAULT_NOTE)).upper(),
                    }
                    for item in data["polygons"]
                    if len(item["vertices"]) >= 3
                ]
                if zones:
                    return zones

            # Support the previous single-polygon format.
            vertices = data["vertices"]
            if len(vertices) >= 3:
                return [{
                    "vertices": [(float(x), float(y)) for x, y in vertices],
                    "note": DEFAULT_NOTE,
                }]

        except (OSError, ValueError, KeyError, TypeError):
            print(f"Could not read {LAYOUT_PATH}; using the default polygon.")

    return [{"vertices": list(DEFAULT_ZONE), "note": DEFAULT_NOTE}]


def save_zones(zones):
    data = {
        "polygons": [
            {"vertices": zone["vertices"], "note": zone["note"]}
            for zone in zones
        ]
    }
    LAYOUT_PATH.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Saved {len(zones)} polygon(s) to {LAYOUT_PATH}")


def add_zone(state):
    state["zones"].append({
        "vertices": [
            (0.40, 0.40),
            (0.60, 0.40),
            (0.60, 0.60),
            (0.40, 0.60),
        ],
        "note": DEFAULT_NOTE,
    })
    state["selected_zone"] = len(state["zones"]) - 1
    state["was_inside"].append(False)
    state["note_channels"].append(None)
    state["note_started_at"].append(None)
    print(f"Added polygon {len(state['zones'])} with note {DEFAULT_NOTE}.")


def handle_mouse(event, x, y, _flags, state):
    width, height = state["size"]

    if event == cv2.EVENT_LBUTTONDOWN:
        nearest_match = None
        nearest_distance = DRAG_RADIUS

        for zone_index, zone in enumerate(state["zones"]):
            for vertex_index, (px, py) in enumerate(zone["vertices"]):
                screen_x, screen_y = int(px * width), int(py * height)
                distance = float(np.hypot(x - screen_x, y - screen_y))

                if distance <= nearest_distance:
                    nearest_distance = distance
                    nearest_match = zone_index, vertex_index

        if nearest_match is not None:
            state["selected_zone"], state["dragging"] = nearest_match

    elif event == cv2.EVENT_MOUSEMOVE and state["dragging"] is not None:
        zone = state["zones"][state["selected_zone"]]
        zone["vertices"][state["dragging"]] = (
            min(max(x / width, 0.0), 1.0),
            min(max(y / height, 0.0), 1.0),
        )

    elif event == cv2.EVENT_LBUTTONUP:
        state["dragging"] = None


def make_notes():
    """Create short note sounds, or return an empty dict if audio is unavailable."""
    try:
        sample_rate = 22050
        pygame.mixer.init(
            frequency=sample_rate,
            size=-16,
            channels=1,
            buffer=1024,
        )

        sample_count = sample_rate // 2
        sample_indices = np.arange(sample_count)
        sounds = {}

        for note_name, frequency in NOTE_FREQUENCIES.items():
            cycles = round(frequency * sample_count / sample_rate)
            samples = (
                0.12
                * 32767
                * np.sin(2 * np.pi * cycles * sample_indices / sample_count)
            ).astype(np.int16)
            sounds[note_name] = pygame.mixer.Sound(buffer=samples.tobytes())

        return sounds

    except pygame.error as error:
        print(f"Audio unavailable; continuing without notes: {error}")
        return {}


def make_zone_polygons(zones, width, height):
    return [
        np.array(
            [(int(x * width), int(y * height)) for x, y in zone["vertices"]],
            dtype=np.int32,
        )
        for zone in zones
    ]


def draw_pose(frame, landmarks, points):
    for start, end in CONNECTIONS:
        if landmarks[start].visibility > 0.5 and landmarks[end].visibility > 0.5:
            cv2.line(frame, points[start], points[end], (0, 255, 0), 2)

    for index, landmark in enumerate(landmarks):
        if landmark.visibility > 0.5:
            cv2.circle(frame, points[index], 4, (0, 0, 255), -1)


def draw_zones(frame, polygons, state):
    for index, polygon in enumerate(polygons):
        selected = index == state["selected_zone"]
        color = (
            (0, 255, 0)
            if state["inside_by_zone"][index]
            else (0, 200, 255)
            if selected
            else (255, 255, 255)
        )

        cv2.polylines(frame, [polygon], True, color, 4)

        for x, y in polygon:
            point = (int(x), int(y))
            cv2.circle(frame, point, 10, color, -1)
            cv2.circle(frame, point, 10, (0, 0, 0), 2)

        label_x, label_y = map(int, polygon[0])
        note = state["zones"][index]["note"]
        cv2.putText(
            frame,
            f"Zone {index + 1}: {note}",
            (label_x, max(20, label_y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )


_zone_debounce = {}

def debounce_zones(raw_inside):
    now = time.monotonic()
    filtered = {}

    for zone_id, is_inside in raw_inside.items():
        state = _zone_debounce.setdefault(
            zone_id,
            {"active": False, "entered_at": None, "exited_at": None},
        )

        if not state["active"]:
            if is_inside:
                if state["entered_at"] is None:
                    state["entered_at"] = now
                elif now - state["entered_at"] >= ENTRY_DWELL:
                    state["active"] = True
                    state["entered_at"] = None
            else:
                state["entered_at"] = None
        else:
            if not is_inside:
                if state["exited_at"] is None:
                    state["exited_at"] = now
                elif now - state["exited_at"] >= EXIT_DWELL:
                    state["active"] = False
                    state["exited_at"] = None
            else:
                state["exited_at"] = None

        filtered[zone_id] = state["active"]

    return filtered


def update_notes(state, person_detected, now, sounds):
    for index, raw_inside in enumerate(state["inside_by_zone"]):
        raw_inside = person_detected and raw_inside
        was_inside = state["was_inside"][index]

        # Debounce entry: require continuous presence for ENTRY_DWELL.
        if not was_inside:
            state["outside_since"][index] = None

            if raw_inside:
                if state["inside_since"][index] is None:
                    state["inside_since"][index] = now
                elif now - state["inside_since"][index] >= ENTRY_DWELL:
                    was_inside = True
                    state["inside_since"][index] = None
            else:
                state["inside_since"][index] = None

        # Debounce exit: require continuous absence for exit_dwell.
        else:
            state["inside_since"][index] = None

            if not raw_inside:
                if state["outside_since"][index] is None:
                    state["outside_since"][index] = now
                elif now - state["outside_since"][index] >= EXIT_DWELL:
                    was_inside = False
                    state["outside_since"][index] = None
            else:
                state["outside_since"][index] = None

        should_play = was_inside
        channel = state["note_channels"][index]

        if should_play and not state["was_inside"][index]:
            sound = sounds.get(state["zones"][index]["note"].upper())
            if sound is not None:
                state["note_channels"][index] = sound.play(loops=-1)
                state["note_started_at"][index] = now

        elif should_play and channel is not None:
            started_at = state["note_started_at"][index]
            if started_at is not None and now - started_at >= 3:
                channel.fadeout(500)
                state["note_channels"][index] = None
                state["note_started_at"][index] = None

        elif not should_play:
            if channel is not None:
                channel.fadeout(500)
            state["note_channels"][index] = None
            state["note_started_at"][index] = None

        state["was_inside"][index] = should_play


def main(daemon=None):
    if daemon is None:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--daemon",
            action="store_true",
            help="Run pose tracking without opening a window",
        )
        daemon = parser.parse_args().daemon

    zones = load_zones()
    sounds = make_notes()

    cap = cv2.VideoCapture(10)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam device 10")

    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Could not read from webcam device 10")

    frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]
    window = "Pose tracking"

    state = {
        "zones": zones,
        "size": (width, height),
        "selected_zone": 0,
        "dragging": None,
        "was_inside": [False] * len(zones),
        "note_channels": [None] * len(zones),
        "note_started_at": [None] * len(zones),
        "inside_by_zone": [False] * len(zones),
        "inside_since": [None] * len(zones),
        "outside_since": [None] * len(zones),
    }

    if not daemon:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.imshow(window, frame)
        cv2.waitKey(1)
        cv2.setMouseCallback(window, handle_mouse, state)

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=2,
        min_pose_detection_confidence=0.8,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    try:
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                polygons = make_zone_polygons(state["zones"], width, height)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(rgb),
                )
                timestamp_ms = time.monotonic_ns() // 1_000_000
                result = landmarker.detect_for_video(image, timestamp_ms)

                state["inside_by_zone"] = [False] * len(state["zones"])

                for landmarks in result.pose_landmarks:
                    points = [
                        (int(landmark.x * width), int(landmark.y * height))
                        for landmark in landmarks
                    ]

                    for zone_index, polygon in enumerate(polygons):
                        for foot_id in FOOT_IDS:
                            if (
                                landmarks[foot_id].visibility > 0.5
                                and cv2.pointPolygonTest(
                                    polygon, points[foot_id], False
                                ) >= 0
                            ):
                                state["inside_by_zone"][zone_index] = True
                                break

                    if not daemon:
                        draw_pose(frame, landmarks, points)

                update_notes(
                    state,
                    person_detected=bool(result.pose_landmarks),
                    now=time.monotonic(),
                    sounds=sounds,
                )

                if not daemon:
                    draw_zones(frame, polygons, state)
                    cv2.putText(
                        frame,
                        "Drag corners | Ctrl+N/N: new polygon | S: save | Q: quit",
                        (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
                    cv2.imshow(window, frame)

                    key = cv2.waitKeyEx(1)
                    if key in (ord("q"), ord("Q"), 27):
                        break
                    if key in (14, ord("n"), ord("N")):
                        add_zone(state)
                    elif key in (ord("s"), ord("S")):
                        save_zones(state["zones"])

    except KeyboardInterrupt:
        pass

    finally:
        cap.release()
        if not daemon:
            cv2.destroyAllWindows()
        pygame.mixer.quit()


if __name__ == "__main__":
    main()
