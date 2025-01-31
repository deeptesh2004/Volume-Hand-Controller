import sys
import time
import math
import threading
from argparse import ArgumentParser
from termcolor import colored
import cv2

import utils
from components import HandDetector, VolumeChanger


class VolumeControlApp:
    def __init__(self, reset_volume: bool):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            sys.exit(colored('Camera could not be opened', 'red'))

        try:
            self.detector = HandDetector()
        except Exception as e:
            sys.exit(colored(f"Failed to initialize HandDetector: {e}", "red"))

        try:
            self.volume = VolumeChanger(reset_volume)
        except Exception as e:
            sys.exit(colored(f"Failed to initialize VolumeChanger: {e}", "red"))

        self.db = self.volume.get_initial_db()
        self.previous_time = time.time()

    def process_frame(self, img):
        """Process each frame to detect hand gestures and adjust volume"""
        # Detect hands and draw landmarks
        self.detector.draw_hands(img)
        lm_positions = self.detector.get_positions(img)

        if lm_positions:
            # Extract thumb and index finger tip positions
            x1, y1 = lm_positions[4][1:3]
            x2, y2 = lm_positions[8][1:3]
            length = math.hypot(x2 - x1, y2 - y1)

            # Draw circles and line for thumb and index finger
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

            # Adjust volume based on distance
            db = self.volume.get_scaled_db(length)
            self.volume.set_volume(length)

        return img

    def display_info(self, img, fps):
        """Display FPS and Volume information on the screen"""
        cv2.putText(img, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, f'Volume: {self.db} dB', (10, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    def run(self):
        """Run the video capture and hand gesture control loop"""
        try:
            while True:
                success, img = self.cap.read()
                if not success:
                    utils.error('Unable to process image', self.cap, self.volume)

                img = self.process_frame(img)

                # Calculate FPS
                current_time = time.time()
                fps = int(1 / (current_time - self.previous_time))
                self.previous_time = current_time

                # Display FPS and volume level
                self.display_info(img, fps)

                # Show the image with overlays
                cv2.imshow('Volume Hand Controller', img)

                # Exit on pressing 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print(colored("\nProgram interrupted. Exiting...", "yellow"))
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        self.volume.reset_default_volume()


def main(reset_volume: bool):
    """Initialize the application and start the video capture"""
    app = VolumeControlApp(reset_volume)
    app.run()


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
        main(reset_volume=args.reset_volume)
    except Exception as e:
        print(colored(f"An error occurred: {e}", "red"))
