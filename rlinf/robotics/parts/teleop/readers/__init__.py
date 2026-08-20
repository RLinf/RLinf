# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Vendor-facing readers for the hardware an operator drives.

Each module here talks to one device -- a serial port, an HID handle, a headset
-- and exposes whatever shape that vendor's SDK gives back. They import no
Gymnasium and no scheduler, so a bench script can drive one directly:

.. code-block:: bash

   python -m rlinf.robotics.parts.teleop.readers.gello --port /dev/ttyUSB0

:mod:`rlinf.robotics.parts.teleop.devices` wraps them as parts, which is what
gives them a lifecycle, placement, and a uniform observation.

Imports are left to callers: each device needs its own vendor package.
"""
