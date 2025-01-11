from typing import List, Tuple

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.python.solutions.hands import Hands


class HandDetector():
    hands: Hands

    
    def __init__(self):
        self.hands = mp.solutions.hands.Hands()
    

    def draw_hands(self, img: np.ndarray) -> None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    img, 
                    hand_landmarks, 
                    mp.solutions.hands.HAND_CONNECTIONS
                )
    

    def get_positions(self, img: np.ndarray) -> List[Tuple[int, int, int]]:
        positions = []
        
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[0]

            for (i, lm) in enumerate(myHand.landmark):
                (height, width, _) = img.shape
                pos_x = int(lm.x * width)
                pos_y = int(lm.y * height)

                positions.append([i, pos_x, pos_y])

        return positions