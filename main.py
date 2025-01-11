import sys
import time
import math
from argparse import ArgumentParser

from termcolor import colored
import cv2

import utils
from components import HandDetector, VolumeChanger


def main(reset_volume: bool) -> None:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        sys.exit(colored('Camera could not be opened', 'red'))

    try:
        detector = HandDetector()
    except Exception as e:
        sys.exit(colored(f"Failed to initialize HandDetector: {e}", "red"))

    try:
        volume = VolumeChanger(reset_volume)
    except Exception as e:
        sys.exit(colored(f"Failed to initialize VolumeChanger: {e}", "red"))

    db = volume.get_initial_db()
    previous_time = time.time()

    while True:
        success, img = cap.read()
        if not success:
            utils.error('Unable to process image', cap, volume)

        # Detect hands and draw landmarks
        detector.draw_hands(img)
        lm_positions = detector.get_positions(img)

        # If a hand is visible, process hand landmarks
        if lm_positions:
            # Landmark 4: Thumb tip; Landmark 8: Index finger tip
            
            x1, y1 = lm_positions[4][1:3]
            x2, y2 = lm_positions[8][1:3]
            length = math.hypot(x2 - x1, y2 - y1)

            # Draw circles and a line between thumb and index finger
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

            # Adjust the volume based on the length
            db = volume.get_scaled_db(length)
            volume.set_volume(length)

        # Measure FPS
        current_time = time.time()
        fps = int(1 / (current_time - previous_time))
        previous_time = current_time

        # Display FPS and volume level
        cv2.putText(img, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, f'Volume: {db} dB', (10, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        # Show the image with overlays
        cv2.imshow('Volume Hand Controller', img)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()
    volume.reset_default_volume()


if __name__ == '__main__':
    parser = ArgumentParser(
        usage='python3 main.py [-r | --reset-volume]',
        description='Volume controller with hand gestures',
        allow_abbrev=False
    )

    parser.add_argument(
        '-r',
        '--reset-volume',
        dest='reset_volume',
        action='store_true',
        help='Restores the system volume to what it was before the script started'
    )
    args = parser.parse_args()

    try:
        main(reset_volume = args.reset_volume)
    except Exception as e:
        print(colored(f"An error occurred: {e}", "red"))
