import can
import logging
import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional

from i2rt.motor_drivers.utils import ReceiveMode


class MailboxListener(can.Listener):
    """Listener that stores incoming CAN frames in per-ID mailboxes."""

    def __init__(
        self,
        mailboxes: Dict[int, Deque[can.Message]],
        condition: threading.Condition,
        maxlen: int = 8,
    ) -> None:
        self._mailboxes = mailboxes
        self._condition = condition
        self._maxlen = maxlen

    def on_message_received(self, msg: can.Message) -> None:
        with self._condition:
            mailbox = self._mailboxes[msg.arbitration_id]
            mailbox.append(msg)
            if self._maxlen and len(mailbox) > self._maxlen:
                mailbox.popleft()
            self._condition.notify_all()

class CanInterface:
    def __init__(
        self,
        channel: str = "PCAN_USBBUS1",
        bustype: str = "socketcan",
        bitrate: int = 1000000,
        name: str = "default_can_interface",
        receive_mode: ReceiveMode = ReceiveMode.p16,
        use_buffered_reader: bool = False,
    ):
        self.bus = can.interface.Bus(bustype=bustype, channel=channel, bitrate=bitrate)
        self.busstate = self.bus.state
        self.name = name
        self.receive_mode = receive_mode
        self.use_buffered_reader = use_buffered_reader
        self._mailboxes: Dict[int, Deque[can.Message]] = defaultdict(deque)
        self._mailbox_condition = threading.Condition()
        self._mailbox_maxlen = 16
        self._response_timeout = 0.02  # seconds

        listeners = []
        self._mailbox_listener = MailboxListener(
            self._mailboxes, self._mailbox_condition, self._mailbox_maxlen
        )
        listeners.append(self._mailbox_listener)

        self.buffered_reader: Optional[can.BufferedReader] = None
        if use_buffered_reader:
            # Initialize BufferedReader for additional debugging/inspection.
            self.buffered_reader = can.BufferedReader()
            listeners.append(self.buffered_reader)

        self.notifier = can.Notifier(self.bus, listeners, timeout=0.001)
        logging.info(
            f"Can interface {self.name} use_buffered_reader: {use_buffered_reader}, "
            f"mailbox_listener_enabled: {bool(self._mailbox_listener)}"
        )

    def close(self) -> None:
        """Shut down the CAN bus."""
        if hasattr(self, "notifier") and self.notifier is not None:
            self.notifier.stop()
        self.bus.shutdown()

    def _send_message_get_response(
        self, id: int, motor_id: int, data: List[int], max_retry: int = 5, expected_id: Optional[int] = None
    ) -> can.Message:
        """Send a message over the CAN bus.

        Args:
            id (int): The arbitration ID of the message.
            data (List[int]): The data payload of the message.

        Returns:
            can.Message: The message that was sent.
        """
        message = can.Message(arbitration_id=id, data=data, is_extended_id=False)
        expected = expected_id
        if expected is None:
            if motor_id is None:
                raise AssertionError("motor_id or expected_id must be provided")
            expected = self.receive_mode.get_receive_id(motor_id)

        self._flush_mailbox(expected)

        for attempt in range(1, max_retry + 1):
            try:
                self.bus.send(message)
                response = self._wait_for_expected_message(expected, self._response_timeout)
                if response:
                    return response
                if attempt >= 5:
                    logging.warning(
                        f"{self.name}: timeout waiting for response id {expected:#x} "
                        f"(attempt {attempt}/{max_retry})"
                    )
            except (can.CanError, AssertionError) as e:
                logging.warning(e)
                logging.warning(
                    "\033[91m"
                    + f"CAN Error {self.name}: Failed to communicate with motor {id} over CAN bus. Retrying..."
                    + "\033[0m"
                )
            time.sleep(0.001)

        raise AssertionError(
            f"fail to communicate with the motor {motor_id} "
            f"(expected can id {expected:#x}) on {self.name} at channel {self.bus.channel_info}"
        )

    def _flush_mailbox(self, arbitration_id: Optional[int]) -> None:
        if arbitration_id is None:
            return
        with self._mailbox_condition:
            mailbox = self._mailboxes.get(arbitration_id)
            if mailbox:
                mailbox.clear()

    def _wait_for_expected_message(self, arbitration_id: int, timeout: float) -> Optional[can.Message]:
        deadline = time.time() + timeout
        with self._mailbox_condition:
            while True:
                mailbox = self._mailboxes.get(arbitration_id)
                if mailbox:
                    return mailbox.popleft()
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self._mailbox_condition.wait(timeout=remaining)

    def _wait_for_any_message(self, timeout: float) -> Optional[can.Message]:
        deadline = time.time() + timeout
        with self._mailbox_condition:
            while True:
                for mailbox in self._mailboxes.values():
                    if mailbox:
                        return mailbox.popleft()
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self._mailbox_condition.wait(timeout=remaining)

    def try_receive_message(self, motor_id: Optional[int] = None, timeout: float = 0.009) -> Optional[can.Message]:
        """Try to receive a message from the CAN bus.

        Args:
            timeout (float): The time to wait for a message (in seconds).

        Returns:
            Optional[can.Message]: The received message, or None if no message is received.
        """
        return self._receive_message(motor_id, timeout, supress_warning=True)

    def _receive_message(
        self, motor_id: Optional[int] = None, timeout: float = 0.009, supress_warning: bool = False
    ) -> Optional[can.Message]:
        """Receive a message from the CAN bus.

        Args:
            timeout (float): The time to wait for a message (in seconds).

        Returns:
            Optional[can.Message]: The received message, or None if timeout happens.
        """
        expected_id: Optional[int] = None
        if motor_id is not None:
            expected_id = self.receive_mode.get_receive_id(motor_id)

        if expected_id is not None:
            message = self._wait_for_expected_message(expected_id, timeout)
        else:
            message = self._wait_for_any_message(timeout)

        if message:
            return message
        if not supress_warning:
            logging.warning(
                "\033[91m"
                + f"Failed to receive message, {self.name} motor id {motor_id} motor timeout. Check if the motor is powered on or if the motor ID exists."
                + "\033[0m"
            )
