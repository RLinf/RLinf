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

"""Bring-up commands for the LeKiwi mobile base and its camera.

The base and the camera connect to the robot's own computer, so these commands
run there and talk to the hardware directly. Install RLinf and run them from
the same environment:

* ``rlinf-lekiwi-test-wheels``: spin each wheel in turn and read it back.
* ``rlinf-lekiwi-check-camera``: open the camera and report the frames it sends.
* ``rlinf-lekiwi-teleoperate``: drive the base with a keyboard, locally or over
  TCP from another machine.
"""
