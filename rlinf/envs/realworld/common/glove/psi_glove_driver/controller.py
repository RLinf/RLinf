#!/usr/bin/env python3
# Copyright (c) 2025 PSI Robot Team
# Licensed under the Apache License, Version 2.0

import logging
import struct
from enum import IntEnum

from .interface import (CommunicationInterface, PSIGloveRequestMessage,
                        PSIGloveRequestType, PSIGloveStatusMessage)


class PSIGloveJointType(IntEnum):
    tip = 0
    mid = 1
    back = 2
    side = 3
    rotate = 4


logger = logging.getLogger(__name__)


class PSIGloveController:

    def __init__(self, communication_interface: CommunicationInterface):
        self.communication_interface = communication_interface
        self.last_status = None

    def connect(self) -> bool:
        return self.communication_interface.connect()

    def disconnect(self):
        return self.communication_interface.disconnect()

    def is_connected(self) -> bool:
        return self.communication_interface.is_connected()

    def read_joint_positions(self) -> PSIGloveStatusMessage:
        request_message = PSIGloveRequestMessage(
            bytes=PSIGloveRequestType.READ_JOINT_POSITION
        )
        response = self.communication_interface.send_and_receive(
            message=request_message
        )
        status = self._parse_response(response)
        return status

    # def read_device_id(self) -> PSIGloveStatusMessage:
    #     request_message = PSIGloveRequestMessage(
    #         bytes=PSIGloveRequestType.READ_GLOVE_ID
    #     )
    #     response = self.communication_interface.send_and_receive(
    #         message=request_message
    #     )
    #     raise NotImplementedError("TODO read_device_id")

    def loop(self) -> PSIGloveStatusMessage:
        status = self.read_joint_positions()
        self.last_status = status
        return status

    def _parse_response(self, raw_bytes: bytes) -> PSIGloveStatusMessage:
        """
        Parse joint position response.
        
        Protocol format:
        - Response header: 01 03 2A (slave address + function code + data length) # noqa
        - Data: 42 bytes joint data (21 joints, 2 bytes each)
        - CRC checksum: 2 bytes
        Joint ordering:
        - Thumb: 5 joints [tip, middle, base, side, rotation]
        - Index finger: 4 joints [tip, middle, base, side]
        - Middle finger: 4 joints [tip, middle, base, side]
        - Ring finger: 4 joints [tip, middle, base, side]
        - Little finger: 4 joints [tip, middle, base, side]
        """
        raw = struct.unpack(">BBB21H", raw_bytes[:45])
        joint_positions = list(raw[3:24])
        
        thumb_joints = joint_positions[0:5]
        index_joints = joint_positions[5:9]
        middle_joints = joint_positions[9:13]
        ring_joints = joint_positions[13:17]
        pinky_joints = joint_positions[17:21]
        
        logger.debug(
            f"Thumb joints: {thumb_joints}, Index joints: {index_joints}, "
            f"Middle joints: {middle_joints}, Ring joints: {ring_joints}, Pinky joints: {pinky_joints}"
        )

        status_message = PSIGloveStatusMessage(
            thumb=thumb_joints,
            index=index_joints,
            middle=middle_joints,
            ring=ring_joints,
            pinky=pinky_joints,
        )
        return status_message
