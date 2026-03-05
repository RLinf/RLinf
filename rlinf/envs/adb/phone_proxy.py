"""
Phone proxy base class.

This module defines a common interface for "phone-like" interaction proxies.
Proxy classes should inherit from this class.
"""

import time
from abc import ABC, abstractmethod
from typing import Optional


class PhoneEnv(ABC):
    """Phone Proxy base class, encapsulating common phone operations"""

    @abstractmethod
    def __init__(self, device_id: Optional[str] = None, adb_path: str = "adb"):
        """
        Initialize Phone Proxy

        Args:
            device_id: Device ID, uses default device if None
            adb_path: ADB executable path, defaults to "adb"
        """
        raise NotImplementedError

    @abstractmethod
    def _run_command(self, command: list[str], timeout: int = 10) -> tuple[bool, str]:
        """
        Execute ADB command

        Args:
            command: Command list
            timeout: Timeout in seconds

        Returns:
            (Success flag, output content)
        """
        raise NotImplementedError

    @abstractmethod
    def _check_device(self) -> bool:
        """
        Check if the device is connected

        Notes:
            - Parses `adb devices` output and checks that a device is in "device" state.
        """
        raise NotImplementedError

    @abstractmethod
    def get_device_info(self) -> dict[str, str]:
        """
        Get device information

        Notes:
            - Collects model, Android version, SDK version, screen resolution and serial number.

        Returns:
            Dictionary containing device information
        """
        raise NotImplementedError

    @abstractmethod
    def get_screen_size(self) -> tuple[int, int]:
        """
        Get screen size

        Notes:
            - Expects `wm size` output format: "Physical size: 1080x1920".

        Returns:
            (Width, Height)
        """
        raise NotImplementedError

    @abstractmethod
    def click(self, x: int, y: int, duration: float = 0.1) -> bool:
        """
        Click at specified location on screen

        Notes:
            - Uses `input tap` under the hood.

        Args:
            x: X coordinate
            y: Y coordinate
            duration: Click duration in seconds, defaults to 0.1s

        Returns:
            Whether successful
        """
        raise NotImplementedError

    @abstractmethod
    def long_press(self, x: int, y: int, duration: float = 1.0) -> bool:
        """
        Long press at specified location on screen

        Notes:
            - Typically implemented via `input swipe x y x y duration_ms`.

        Args:
            x: X coordinate
            y: Y coordinate
            duration: Long press duration in seconds, defaults to 1.0s

        Returns:
            Whether successful
        """
        raise NotImplementedError

    @abstractmethod
    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration: float = 0.5) -> bool:
        """
        Swipe operation

        Notes:
            - Duration is usually passed in milliseconds to the underlying implementation.

        Args:
            x1: Start X coordinate
            y1: Start Y coordinate
            x2: End X coordinate
            y2: End Y coordinate
            duration: Swipe duration in seconds, defaults to 0.5s

        Returns:
            Whether successful
        """
        raise NotImplementedError

    @abstractmethod
    def input_text(self, text: str) -> bool:
        """
        Input text

        Notes:
            - Implementations may need to escape spaces/special chars for shell/adb.

        Args:
            text: Text to input

        Returns:
            Whether successful
        """
        raise NotImplementedError

    @abstractmethod
    def input_keyevent(self, keycode: int) -> bool:
        """
        Send key event

        Args:
            keycode: Key code (e.g., 4=Back, 3=Home, 66=Enter)

        Returns:
            Whether successful
        """
        raise NotImplementedError

    @abstractmethod
    def back(self) -> bool:
        """Back key"""
        raise NotImplementedError

    @abstractmethod
    def home(self) -> bool:
        """Home key"""
        raise NotImplementedError

    @abstractmethod
    def menu(self) -> bool:
        """Menu key"""
        raise NotImplementedError

    @abstractmethod
    def screenshot(self, save_path: str = "screenshot.png") -> bool:
        """
        Screenshot

        Notes:
            - A common pattern is to `screencap` to a temporary path on device, then `pull`,
              and finally cleanup the temp file.

        Args:
            save_path: Save path

        Returns:
            Whether successful
        """
        raise NotImplementedError

    @abstractmethod
    def install_app(self, apk_path: str, replace: bool = True) -> bool:
        """
        Install app

        Args:
            apk_path: APK file path
            replace: Whether to replace existing app

        Returns:
            Whether successful
        """
        raise NotImplementedError

    @abstractmethod
    def uninstall_app(self, package_name: str) -> bool:
        """
        Uninstall app

        Args:
            package_name: Package name

        Returns:
            Whether successful
        """
        raise NotImplementedError

    @abstractmethod
    def start_app(self, package_name: str, activity_name: Optional[str] = None) -> bool:
        """
        Start app

        Args:
            package_name: Package name
            activity_name: Activity name, starts main Activity if None

        Returns:
            Whether successful
        """
        raise NotImplementedError

    @abstractmethod
    def stop_app(self, package_name: str) -> bool:
        """
        Stop app

        Args:
            package_name: Package name

        Returns:
            Whether successful
        """
        raise NotImplementedError

    @abstractmethod
    def get_current_activity(self) -> Optional[str]:
        """
        Get current Activity

        Notes:
            - Implementations often parse `dumpsys window windows` and extract `mCurrentFocus`.

        Returns:
            Activity name, returns None if failed
        """
        raise NotImplementedError

    @abstractmethod
    def get_installed_packages(self) -> list[str]:
        """
        Get list of installed package names

        Returns:
            Package name list
        """
        raise NotImplementedError

    @abstractmethod
    def push_file(self, local_path: str, remote_path: str) -> bool:
        """
        Push file to device

        Args:
            local_path: Local file path
            remote_path: Remote path on device

        Returns:
            Whether successful
        """
        raise NotImplementedError

    @abstractmethod
    def pull_file(self, remote_path: str, local_path: str) -> bool:
        """
        Pull file from device

        Args:
            remote_path: Remote path on device
            local_path: Local save path

        Returns:
            Whether successful
        """
        raise NotImplementedError

    @abstractmethod
    def shell_command(self, command: str) -> tuple[bool, str]:
        """
        Execute shell command

        Notes:
            - Implementations may use `shlex.split` to properly handle quoting.

        Args:
            command: Shell command (can contain spaces and special characters)

        Returns:
            (Whether successful, output content)
        """
        raise NotImplementedError

    @abstractmethod
    def wait_for_device(self, timeout: int = 30) -> bool:
        """
        Wait for device connection

        Args:
            timeout: Timeout in seconds

        Returns:
            Whether successful
        """
        raise NotImplementedError

    @abstractmethod
    def reboot(self, mode: str = "system") -> bool:
        """
        Reboot device

        Args:
            mode: Reboot mode ("system", "bootloader", "recovery")

        Returns:
            Whether successful
        """
        raise NotImplementedError

    def execute_mobile_action(self, action: str, **kwargs) -> bool:
        """
        Execute mobile action

        Args:
            action: Action name
            **kwargs: Keyword arguments
        """
        if action == "click":
            # Click at coordinate (x, y)
            coordinate = kwargs.get("coordinate")
            if not coordinate or len(coordinate) != 2:
                raise ValueError("click action requires coordinate [x, y]")
            x, y = coordinate
            return self.click(x, y)

        elif action == "long_press":
            # Long press at coordinate (x, y) for specified seconds
            coordinate = kwargs.get("coordinate")
            time_duration = kwargs.get("time", 1.0)
            if not coordinate or len(coordinate) != 2:
                raise ValueError("long_press action requires coordinate [x, y]")
            x, y = coordinate
            return self.long_press(x, y, duration=time_duration)

        elif action == "swipe":
            # Swipe from (x, y) to (x2, y2)
            coordinate = kwargs.get("coordinate")
            coordinate2 = kwargs.get("coordinate2")
            if not coordinate or len(coordinate) != 2:
                raise ValueError("swipe action requires coordinate [x, y]")
            if not coordinate2 or len(coordinate2) != 2:
                raise ValueError("swipe action requires coordinate2 [x2, y2]")
            x1, y1 = coordinate
            x2, y2 = coordinate2
            return self.swipe(x1, y1, x2, y2)

        elif action == "type":
            # Input text
            text = kwargs.get("text")
            if not text:
                raise ValueError("type action requires text parameter")
            return self.input_text(text)

        elif action == "answer":
            # Return answer (not an ADB operation)
            text = kwargs.get("text", "")
            return f"Answer: {text}"

        elif action == "system_button":
            # Press system button (Back, Home, Menu, Enter)
            button = kwargs.get("button")
            if not button:
                raise ValueError("system_button action requires button parameter")

            button_map = {
                "Back": self.back,
                "Home": self.home,
                "Menu": self.menu,
                "Enter": self.input_keyevent(66),
            }

            if button not in button_map:
                raise ValueError(
                    f"Unknown button: {button}. Must be Back, Home, Menu, or Enter"
                )

            return button_map[button]()

        elif action == "wait":
            # Wait for specified seconds
            time_duration = kwargs.get("time")
            if time_duration is None:
                raise ValueError("wait action requires time parameter")
            time.sleep(time_duration)
            return True

        elif action == "terminate":
            # Terminate task (not an ADB operation)
            status = kwargs.get("status", "success")
            return f"Task terminated with status: {status}"

        raise ValueError(f"Unknown action: {action}")
