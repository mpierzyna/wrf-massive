"""
Set up yaml serializer to read and write config files (based on ``pydantic.BaseModel``).
"""

from __future__ import annotations

import datetime
import pathlib
from typing import Dict, Any, Self

try:
    import isodate

    ISODATE = True
except ImportError:
    ISODATE = False

import pydantic
import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def path_representer(dumper: yaml.Dumper, path: pathlib.Path):
    """Represent ``pathlib.Path`` as string in yaml."""
    return dumper.represent_scalar(tag="!path", value=str(path))


def path_constructor(loader: yaml.loader, node) -> pathlib.Path:
    """Convert string back to ``pathlib.Path`` object."""
    return pathlib.Path(loader.construct_scalar(node))


def timedelta_representer(dumper: yaml.Dumper, td: datetime.timedelta):
    """Represent ``datetime.timedelta`` as ISO 8601 duration string with yaml tag ``!timedelta``."""
    if not ISODATE:
        raise ImportError("Please install isodate to use timedelta_representer.")
    return dumper.represent_scalar(tag="!timedelta", value=isodate.duration_isoformat(td))


def timedelta_constructor(loader: yaml.Loader, node) -> datetime.timedelta:
    """Convert seconds back to ``timedelta`` object.
    Attention! Changing constructors might result in old files becoming unreadable!
    """
    if not ISODATE:
        raise ImportError("Please install isodate to use timedelta_representer.")
    return isodate.parse_duration(loader.construct_scalar(node))


def tuple_representer(dumper: yaml.Dumper, t: tuple):
    """Convert tuple to yaml list. Attention! Deserialisation will be list not tuple! Pydantic will fix that."""
    return dumper.represent_sequence("tag:yaml.org,2002:seq", t, flow_style=True)


# Register path representer and constructor for Posix and Windows to Dumper
yaml.add_representer(pathlib.Path, path_representer, Dumper=Dumper)
yaml.add_representer(pathlib.PosixPath, path_representer, Dumper=Dumper)
yaml.add_representer(pathlib.WindowsPath, path_representer, Dumper=Dumper)
yaml.add_constructor("!path", path_constructor, Loader=Loader)

# Register representer and constructor to convert timedelta between Python and yaml.
yaml.add_representer(datetime.timedelta, timedelta_representer, Dumper=Dumper)
yaml.add_constructor("!timedelta", timedelta_constructor, Loader=Loader)

# Register representer and constructor to convert tuple between Python and yaml.
yaml.add_representer(tuple, tuple_representer, Dumper=Dumper)


def yaml_to_dict(yaml_str: str) -> Dict:
    """Convert yaml string to dict."""
    return yaml.load(yaml_str, Loader=Loader)


def dict_to_yaml(d: Dict) -> str:
    """Convert dict to yaml string."""
    return yaml.dump(d, Dumper=Dumper)


class BaseYAMLConfig(pydantic.BaseModel):
    """Mixin to add yaml dumping and loading support to pydantic ``BaseModel``.
    Following pydantic v2 paradigm.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)  # Allow, e.g., numpy arrays or custom types

    def model_dump_yaml(self, *, exclude: Dict[str, Any] = None) -> str:
        """Convert model to yaml string."""
        if exclude is None:
            exclude = {}
        return yaml.dump(self.model_dump(exclude=exclude), Dumper=Dumper)

    @classmethod
    def model_from_yaml(cls, yaml_str: str) -> Self:
        """Load model from yaml string."""
        return cls(**yaml_to_dict(yaml_str))  # noqa: unexpected arguments
