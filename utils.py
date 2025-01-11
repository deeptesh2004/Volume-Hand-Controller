import sys

import cv2
from termcolor import colored
from components import VolumeChanger


def error(message: str, cap: cv2.VideoCapture, volume: VolumeChanger) -> None:
    volume.reset_default_volume()
    cap.release()
    cv2.destroyAllWindows()

    sys.exit(colored(message, 'red'))