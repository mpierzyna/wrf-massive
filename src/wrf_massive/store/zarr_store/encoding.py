# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Per-variable encodings of the zarr store: what precision a variable keeps (see DESIGN.md, Encoding).

The models only describe the encoding; `codecs()` maps them to zarr codecs (imported lazily, so configs load without
the `store` extra). Data is always float32 on disk; the compressor follows from the encoding.
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal

import numpy as np
import pydantic

INT32 = np.iinfo(np.int32)


class _Encoding(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    lossy: ClassVar[bool] = True

    def filters(self) -> list:
        return []

    def check(self, name: str, values: np.ndarray) -> None:
        """Raise if `values` cannot be encoded as configured."""

    def codecs(self) -> dict:
        """`filters`/`compressors` arguments of `zarr.create_array`."""
        from zarr.codecs import BloscCodec

        # Byte shuffle compresses unchanged float32 best; bitshuffle is better once low bits are zeroed or the data
        # is integer (benchmark in DESIGN.md)
        shuffle = "bitshuffle" if self.lossy else "shuffle"
        return dict(filters=self.filters() or None, compressors=[BloscCodec(cname="zstd", clevel=3, shuffle=shuffle)])


class Lossless(_Encoding):
    """No precision lost (default for variables without an encoding)."""

    kind: Literal["lossless"] = "lossless"
    lossy: ClassVar[bool] = False


class BitRound(_Encoding):
    """Keep `keepbits` mantissa bits: relative error ≤ 2^-(keepbits+1) of the value, at every magnitude."""

    kind: Literal["bitround"] = "bitround"
    keepbits: int = pydantic.Field(ge=1, le=23)

    def filters(self) -> list:
        from zarr.codecs.numcodecs import BitRound as _BitRound

        return [_BitRound(keepbits=self.keepbits)]


class FixedScaleOffset(_Encoding):
    """Store `round((x - offset) / step)` as int32: absolute error ≤ step / 2, e.g. `offset=300, step=1e-3` for θ."""

    kind: Literal["fixed_scale_offset"] = "fixed_scale_offset"
    offset: float
    step: float = pydantic.Field(gt=0)

    def filters(self) -> list:
        from zarr.codecs.numcodecs import FixedScaleOffset as _FixedScaleOffset

        return [_FixedScaleOffset(offset=self.offset, scale=1 / self.step, dtype="<f4", astype="<i4")]

    def check(self, name: str, values: np.ndarray) -> None:
        # numcodecs casts to int32 without checks: NaN and out-of-range values would silently wrap around
        if not np.all(np.isfinite(values)):
            raise ValueError(f"'{name}': FixedScaleOffset cannot store NaN or inf.")
        q_min = (float(values.min()) - self.offset) / self.step
        q_max = (float(values.max()) - self.offset) / self.step
        if q_min < INT32.min or q_max > INT32.max:
            raise ValueError(
                f"'{name}': values {values.min()}..{values.max()} do not fit int32 with offset={self.offset}, "
                f"step={self.step} (representable: {self.offset + INT32.min * self.step:.6g}.."
                f"{self.offset + INT32.max * self.step:.6g})."
            )


Encoding = Annotated[Lossless | BitRound | FixedScaleOffset, pydantic.Field(discriminator="kind")]
