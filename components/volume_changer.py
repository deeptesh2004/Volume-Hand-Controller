import sys

import numpy as np
from termcolor import colored
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

from constants import VOLUME_DISTANCE_MIN, VOLUME_DISTANCE_MAX


class VolumeChanger():
    reset_volume: bool
    volume: IAudioEndpointVolume
    initial_volume: int
    min_volume: int
    max_volume: int


    def __init__(self, reset_volume: bool):
        self.reset_volume = reset_volume

        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(
            IAudioEndpointVolume._iid_, 
            CLSCTX_ALL, 
            None
        )
        self.volume = interface.QueryInterface(IAudioEndpointVolume)

        if self.volume.GetMute() == 1:
            sys.exit(colored(
                'Unmute the volume and run the script again', 
                'red'
            ))
        
        self.initial_volume = int(self.volume.GetMasterVolumeLevel())
        (self.min_volume, self.max_volume, _) = self.volume.GetVolumeRange()
        self.vol = self.volume.GetMasterVolumeLevel()
    

    def get_initial_db(self) -> int:
        return self.initial_volume
    

    def get_scaled_db(self, length: int) -> int:
        return int(
            np.interp(
                length, 
                [VOLUME_DISTANCE_MIN, VOLUME_DISTANCE_MAX], 
                [self.min_volume, self.max_volume]
            )
        )
    

    def set_volume(self, length: int) -> None:
        new_volume = np.interp(
            length,
            [VOLUME_DISTANCE_MIN, VOLUME_DISTANCE_MAX],
            [self.min_volume, self.max_volume]
        )

        self.volume.SetMasterVolumeLevel(new_volume, None)
    

    def reset_default_volume(self) -> None:
        if self.reset_volume:
            self.volume.SetMasterVolumeLevel(self.initial_volume, None)