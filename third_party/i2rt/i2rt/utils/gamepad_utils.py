#!/usr/bin/env python3
from evdev import InputDevice, categorize, ecodes, list_devices
import math
import pygame
import os
import numpy as np
import threading

class Gamepad:
    def __init__(self):
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        # Initialize pygame and joystick
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            print("No joystick/gamepad connected!")
            exit()
        else:
            print(f"Detected {pygame.joystick.get_count()} joystick(s).")

        # Initialize the joystick
        self.joy = pygame.joystick.Joystick(0)
        self.joy.init()

        print(f"Joystick Name: {self.joy.get_name()}")
        print(f"Number of Axes: {self.joy.get_numaxes()}")
        print(f"Number of Buttons: {self.joy.get_numbuttons()}")

    def get_button_reading(self):
        pygame.event.pump()
        key_mode = self.joy.get_button(12)
        key_left_2 = self.joy.get_button(8)
        key_left_1 = self.joy.get_button(6)
        return dict(
            key_mode=key_mode,
            key_left_2=key_left_2,
            key_left_1=key_left_1,
        )

    def get_user_cmd(self):
        pygame.event.pump()
        # Read inputs
        x = self.joy.get_axis(1)  # Left stick Y-axis
        y = self.joy.get_axis(0)  # Left stick X-axis
        th = self.joy.get_axis(2)  # Right stick X-axis

        user_cmd = np.array([-x, y, th])
        # if < 0.05 force to zero
        user_cmd[np.abs(user_cmd) < 0.05] = 0
        return user_cmd

    def close(self):
        pygame.quit()



if __name__ == "__main__":
    pad = Gamepad()
    try:
        while True:
            cmd = pad.get_user_cmd()
            print(f"user_cmd = {cmd}")
    except KeyboardInterrupt:
        pad.close()
