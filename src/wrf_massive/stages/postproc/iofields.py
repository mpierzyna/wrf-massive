# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Reading WRF iofields files (`myoutfields.txt`)."""

import pathlib
from typing import Set


def parse_myoutfields(path: pathlib.Path | str, stream: int) -> Set[str]:
    """Upper-cased names of the WRF fields that an iofields file (`myoutfields.txt`) adds to history stream `stream`.

    Lines look like `+:h:7:U,V,T` (add `U`, `V`, `T` to history stream 7) or `-:h:0:...` (remove from stream 0).
    The file is read like WRF's `modify_io_masks` (`frame/module_domain.F`) reads it: a line containing `#` anywhere
    is a comment and reading stops at the first blank line. WRF matches the names case-insensitively and writes them
    upper-case, hence the upper-casing. Lines for input streams (`i`) or other history streams are ignored.
    """
    fields: Set[str] = set()
    for lineno, line in enumerate(pathlib.Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            break  # WRF stops reading at the first blank line
        if "#" in line:
            continue
        parts = [p.strip() for p in line.split(":", 3)]
        if len(parts) != 4 or parts[0] not in ("+", "-"):
            raise ValueError(f"{path}, line {lineno}: expected '<+|->:<h|i>:<stream>:<fields>', got '{line}'.")
        op, stream_type, stream_id, names = parts
        if stream_type.lower() != "h" or int(stream_id) != stream:
            continue
        names = {n.strip().upper() for n in names.split(",") if n.strip()}
        fields = fields | names if op == "+" else fields - names
    return fields
