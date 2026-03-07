#!/usr/bin/env python3
# Copyright (c) 2025 PSI Robot Team
# Licensed under the Apache License, Version 2.0

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


class PSIGloveRequestType:
    READ_JOINT_POSITION = bytes.fromhex("010300010015D5C5")
    READ_GLOVE_ID = bytes.fromhex("01030007001535C4")


@dataclass
class PSIGloveRequestMessage:
    bytes: bytes


@dataclass
class PSIGloveStatusMessage:
    thumb: List[int]
    index: List[int]
    middle: List[int]
    ring: List[int]
    pinky: List[int]


class CommunicationInterface(ABC):

    def __init__(self, auto_connect: bool):
        self.connected = False
        if auto_connect:
            self.connect()

    @abstractmethod
    def connect(self) -> bool:
        pass

    @abstractmethod
    def disconnect(self):
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        pass

    @abstractmethod
    def send_and_receive(self, message: PSIGloveRequestMessage) -> bytes:
        pass


class SerialInterface(CommunicationInterface):

    def __init__(
        self,
        port: str,
        baudrate: int,
        timeout: float = 0.006,
        auto_connect: bool = False,
        mock: bool = False,
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_controller = None
        self.mock = mock
        super().__init__(auto_connect)

    def connect(self) -> bool:
        if self.mock:
            self.connected = True
            return True
        try:
            import serial

            self.serial_controller = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
            )
            self.connected = True
            logger.info(f"Serial port opened successfully: {self.port}")
            return True
        except ImportError:
            self.connected = False
            logger.error(
                "Serial library not installed, please install pyserial: \
                    pip install pyserial"
            )
            return False
        except Exception as e:
            self.connected = False
            logger.error(f"Serial port open failed: {e}")
            return False

    def disconnect(self):
        if self.serial_controller and self.connected:
            self.serial_controller.close()
            self.connected = False
            logger.info("Serial connection disconnected")

    def is_connected(self) -> bool:
        return self.connected

    def _send_message(self, message: PSIGloveRequestMessage) -> bool:
        if not self.connected:
            logger.error("Serial port not connected")
            return False

        if self.mock:
            logger.debug(
                f"Send - Frame data: {[hex(x) for x in message.bytes]}"
            )
            return True

        try:
            self.serial_controller.write(message.bytes)
            logger.debug(
                f"Send - Frame data: {[hex(x) for x in message.bytes]}"
            )
            return True
        except Exception as e:
            logger.error(f"Serial message send failed: {e}")
            return False

    def _receive_message(self) -> bytes:
        if not self.connected:
            return None
        try:
            response = self.serial_controller.read(64)
            return response
        except Exception as e:
            logger.error(f"Serial message receive failed: {e}")
            return None

    def send_and_receive(self, message: PSIGloveRequestMessage) -> bytes:
        if not self._send_message(message):
            return {}

        if self.mock:
            logger.debug("Mock mode, no need to receive data")
            return {}

        response = self._receive_message()
        return response
