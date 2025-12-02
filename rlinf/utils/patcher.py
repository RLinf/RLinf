# Copyright 2025 The RLinf Authors.
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

import importlib
import inspect
import sys
import types
from typing import Callable


class _Patcher:
    def __init__(self):
        self._mappings_dict: dict[str, str] = {}
        self._wrappers_dict: dict[str, list[Callable]] = {}

    def clear(self):
        self._mappings_dict = {}
        self._wrappers_dict = {}
        self._mappings = {}
        self._traced_module = set()
        self._traced_func = set()
        self._traced_cls = set()

    @staticmethod
    def _get_parent_obj_and_obj(name: str):
        name_list = name.split(".")
        if len(name_list) == 1:
            try:
                curr_obj = importlib.import_module(name)
                return None, curr_obj
            except ModuleNotFoundError:
                return None, None
        obj_list = []
        for i in range(1, len(name_list) + 1):
            # parent = ".".join(name_list[: i - 1])
            path = ".".join(name_list[:i])
            try:
                curr_obj = importlib.import_module(path)
                obj_list.append(curr_obj)
            except ModuleNotFoundError:
                if i == 1:
                    raise RuntimeError(f"prefix object not found in {name}")

                for j in range(i - 2, len(name_list) - 2):
                    if hasattr(obj_list[j], name_list[j + 1]):
                        obj_list.append(getattr(obj_list[j], name_list[j + 1]))
                    else:
                        raise RuntimeError(f"prefix object not found in {name}")
                if hasattr(obj_list[-1], name_list[-1]):
                    return obj_list[-1], getattr(obj_list[-1], name_list[-1])
                else:
                    return obj_list[-1], None
        return obj_list[-2], obj_list[-1]

    def _parse_mappings(self):
        # parse all objs and build self._mappings
        self._mappings = {}
        for old, new in self._mappings_dict.items():
            new_parent_obj, new_obj = self._get_parent_obj_and_obj(new)
            if new_obj is None:
                raise RuntimeError(f"object not exist: {new}")
            if old in self._wrappers_dict:
                for wrapper in self._wrappers_dict[old]:
                    new_obj = wrapper(new_obj)
            old_parent_obj, old_obj = self._get_parent_obj_and_obj(old)

            if new_parent_obj is None or old_parent_obj is None:
                assert inspect.ismodule(new_obj), f"new object is not a module: {new}"

            # When the old_parent_object is a class, check whether the old object is actually a patched function from the *parent class* (not the current class).
            # If so, set old_obj to None to avoid triggering the repatch assertion below.
            if old_obj is not None:
                if inspect.isclass(old_parent_obj):
                    attr_name = old.split(".")[-1]
                    # Use super to get the parent class
                    parent_attr = getattr(
                        super(old_parent_obj, old_parent_obj), attr_name, None
                    )
                    if parent_attr == old_obj:
                        old_obj = None

            if old_obj is None:
                # create temparary dummy object, which will be replaced by new_obj in apply stage
                from unittest.mock import Mock

                old_obj = Mock(
                    side_effect=KeyError(
                        "patcher internal error, dummy object not replaced"
                    )
                )
                if inspect.ismodule(new_obj):
                    sys.modules[old] = old_obj
                if old_parent_obj is not None:
                    setattr(old_parent_obj, old.split(".")[-1], old_obj)

            assert id(old_obj) not in self._mappings, (
                f"do not support re_patch! old object is [{repr(old_obj)}], news objects are [{repr(self._mappings[id(old_obj)])}] and [{repr(new_obj)}]"
            )
            self._mappings[id(old_obj)] = new_obj

    def _apply_to_class(self, cls):
        # patch member functions and member classes in classes
        if cls in self._traced_cls:
            return
        self._traced_cls.add(cls)

        for k, v in cls.__dict__.items():
            # most function with prefix '__' means it's an inner operator or variable
            # that should no be patched. We will not patch them execpt __init__ which
            # is frequently used and should be patched.
            if k.startswith("ORIG__"):
                continue

            if inspect.isclass(v):
                Patcher._apply_to_class(v)

            patch_target_obj = None
            original_id = -1

            if isinstance(v, staticmethod):
                # If it's a staticmethod descriptor, get the underlying function
                patch_target_obj = v.__func__
                original_id = id(patch_target_obj)
            elif isinstance(v, classmethod):
                # If it's a classmethod descriptor, get the underlying function
                patch_target_obj = v.__func__
                original_id = id(patch_target_obj)
            else:
                # For regular methods or other attributes
                patch_target_obj = v
                original_id = id(patch_target_obj)

            if original_id in self._mappings:
                new_obj = self._mappings[original_id]

                if id(cls) in self._mappings:
                    raise RuntimeError(
                        f"Patcher: cannot patch a class and the attr in this class meanwhile! "
                        f"cls is [{repr(cls)}], attr is [{repr(v)}]"
                    )

                # Re-wrap if necessary
                if isinstance(v, staticmethod):
                    setattr(cls, k, staticmethod(new_obj))
                elif isinstance(v, classmethod):
                    setattr(cls, k, classmethod(new_obj))
                else:
                    setattr(cls, k, new_obj)

    def _apply_to_modules(self):
        self._traced_module = set()
        self._traced_func = set()
        self._traced_cls = set()
        keys_to_patch = []
        for key, value in sys.modules.copy().items():
            if id(value) in self._mappings:
                keys_to_patch.append(key)
                sys.modules[key] = self._mappings[id(value)]

        for key in keys_to_patch:
            for k, v in sys.modules.copy().items():
                if k.startswith(key) and k != key:
                    del sys.modules[k]

        for key, value in sys.modules.copy().items():
            for k, v in value.__dict__.copy().items():
                if k.startswith("ORIG__"):
                    continue

                if inspect.isclass(v):
                    self._apply_to_class(v)

                if id(v) in self._mappings:
                    setattr(value, k, self._mappings[id(v)])

    def add_patch(self, old: str, new: str):
        assert isinstance(old, str)
        assert isinstance(new, str)
        if old in self._mappings_dict:
            raise RuntimeError(
                f"do not support re_patch! old object is [{old}], new objets are [{self._mappings_dict[old]}] and [{new}]"
            )
        self._mappings_dict[old] = new

    def add_wrapper(self, old: str, wrapper: types.FunctionType):
        assert isinstance(old, str)
        if wrapper is None:
            raise RuntimeError(f"wrapper is None in add_wrapper, old is [{old}]")
        assert callable(wrapper)
        if old not in self._wrappers_dict:
            self._wrappers_dict[old] = []
        self._wrappers_dict[old].append(wrapper)

    def apply(self):
        for old in self._wrappers_dict:
            if old not in self._mappings_dict:
                self._mappings_dict[old] = old

        if len(self._mappings_dict) == 0:
            return

        self._parse_mappings()
        self._apply_to_modules()


Patcher = _Patcher()
