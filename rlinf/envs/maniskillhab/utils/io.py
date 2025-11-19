# modified from https://stackoverflow.com/a/42711281

import json
import re

from _ctypes import PyObj_FromPtr  # see https://stackoverflow.com/a/15012814/355230

from mani_skill.utils.io_utils import CustomJsonEncoder


class NoIndent(object):
    """Value wrapper."""

    def __init__(self, value):
        if not isinstance(value, (list, tuple)):
            raise TypeError("Only lists and tuples can be wrapped")
        self.value = value


class NoIndentSupportingJSONEncoder(CustomJsonEncoder):
    FORMAT_SPEC = "@@{}@@"  # Unique string pattern of NoIndent object ids.
    regex = re.compile(FORMAT_SPEC.format(r"(\d+)"))  # compile(r'@@(\d+)@@')

    def __init__(self, **kwargs):
        # Keyword arguments to ignore when encoding NoIndent wrapped values.
        ignore = {"cls", "indent"}

        # Save copy of any keyword argument values needed for use here.
        self._kwargs = {k: v for k, v in kwargs.items() if k not in ignore}
        super().__init__(**kwargs)

    def default(self, obj):
        return (
            self.FORMAT_SPEC.format(id(obj))
            if isinstance(obj, NoIndent)
            else super().default(obj)
        )

    def iterencode(self, obj, **kwargs):
        format_spec = self.FORMAT_SPEC  # Local var to expedite access.

        # Replace any marked-up NoIndent wrapped values in the JSON repr
        # with the json.dumps() of the corresponding wrapped Python object.
        for encoded in super().iterencode(obj, **kwargs):
            match = self.regex.search(encoded)
            if match:
                id = int(match.group(1))
                no_indent = PyObj_FromPtr(id)
                json_repr = json.dumps(
                    no_indent.value, cls=CustomJsonEncoder, **self._kwargs
                )
                # Replace the matched id string with json formatted representation
                # of the corresponding Python object.
                encoded = encoded.replace(
                    '"{}"'.format(format_spec.format(id)), json_repr
                )

            yield encoded
