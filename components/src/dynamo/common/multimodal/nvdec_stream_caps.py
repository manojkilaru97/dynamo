# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-decode NVDEC safety check for the Nemotron Omni video path.

PyNvVideoCodec 2.2.x segfaults the whole worker process, instead of raising,
on the first decoded frame of:

* VP9 profile 1 or 3 (4:2:2, 4:4:0, 4:4:4), in any container. The image's
  libavformat ships no codec parsers, so the demuxer reports 4:2:0 8-bit for
  every stream and builds a 4:2:0 decoder.
* VP9 read from ISO-BMFF (MP4/MOV), even at profile 0.

Omni routes VP8/VP9/AV1 to NVDEC, so one such upload took down a worker, and
request migration plus client retries carried it across the fleet.

The check asks PyNvVideoCodec's own demuxer -- the same libavformat code path
the decoder reads through -- for the stream's codec and container, and for
VP9 inspects the uncompressed header of every frame in every packet it yields
(each frame of a superframe included). It never creates a decoder, so it
cannot hit the crash. Rules:

* VP9: Matroska/WebM only, every frame profile 0 or 2 (4:2:0).
* VP8: Matroska/WebM only (VP8/VP9 from MP4 is outside what was measured safe).
* AV1: refused. NVDEC never decodes AV1 through PyNvVideoCodec 2.2.x on these
  images (it raises), so this only skips a doomed attempt.
* Everything else (H.264, HEVC, ...): unchanged routing.

With ``strict`` (a VP8/VP9/AV1 byte-marker hint) a stream the demuxer cannot
open is refused; without it (an H.264/HEVC hint) it is left to the decoder,
as before. Only the Omni loader imports this module.
"""

from __future__ import annotations

import ctypes
import tempfile

_NVDEC_VP9_PROFILES = frozenset({0, 2})
# About 1.8 hours at 30 fps; bounds the work a crafted input can cause.
_MAX_PACKETS = 200_000
# A zero-length read is not proof of end of stream (an empty block, or a read
# error the demuxer resyncs past), so keep reading until several in a row.
_EOF_EMPTY_READS = 3


class _Unsupported(Exception):
    """A condition NVDEC must not see."""


def unsupported_reason(data: bytes, strict: bool) -> str | None:
    """Why NVDEC must not decode ``data``, or ``None`` if it may."""
    import PyNvVideoCodec as nvc

    # The demuxer takes a path, like the decoder in ``decode_video_nvdec``.
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp.write(data)
        tmp.flush()
        try:
            demuxer = nvc.CreateDemuxer(tmp.name)
            codec = demuxer.GetNvCodecId()
            container = demuxer.GetContainerName() or ""
        except Exception as exc:  # noqa: BLE001 - unreadable input is not fatal here
            return (
                f"demuxer could not open the stream ({type(exc).__name__})"
                if strict
                else None
            )
        if codec == nvc.cudaVideoCodec.AV1:
            return "AV1 is not decodable through PyNvVideoCodec on this image"
        if codec not in (nvc.cudaVideoCodec.VP9, nvc.cudaVideoCodec.VP8):
            return None
        if not container.startswith("matroska"):
            name = "VP9" if codec == nvc.cudaVideoCodec.VP9 else "VP8"
            where = container or "an unknown container"
            return f"{name} in {where}; decoded on NVDEC only from Matroska/WebM"
        if codec == nvc.cudaVideoCodec.VP8:
            return None
        try:
            frames = _check_vp9_packets(demuxer)
        except _Unsupported as exc:
            return str(exc)
        except Exception as exc:  # noqa: BLE001
            return f"VP9 stream could not be inspected ({type(exc).__name__})"
        return None if frames else "no VP9 frame found"


def _check_vp9_packets(demuxer) -> int:
    """Check every frame header of every packet the demuxer yields."""
    frames = empty = 0
    for _ in range(_MAX_PACKETS):
        packet = demuxer.Demux()
        size = packet.bsl
        if not size:
            empty += 1
            if empty >= _EOF_EMPTY_READS:
                return frames
            continue
        empty = 0
        frames += check_vp9_packet(ctypes.string_at(packet.bsl_data, size))
    raise _Unsupported("VP9 stream too long to inspect")


def check_vp9_packet(data: bytes) -> int:
    """Check each frame header in one VP9 packet; return the frame count.

    A superframe ends with an index whose first and last bytes are the same
    marker 0b110xxxxx (VP9 spec, Annex B). libvpx walks such an index entry by
    entry and decodes every frame that fits, so an index whose sizes do not
    fit is refused rather than read as one frame. Encoders pad any frame whose
    last byte merely looks like a marker, so valid streams never hit this.
    """
    if not data:
        raise _Unsupported("empty VP9 packet")
    last = data[-1]
    if last & 0xE0 == 0xC0:
        count = (last & 0x7) + 1
        width = ((last >> 3) & 0x3) + 1
        index_len = 2 + width * count
        if len(data) >= index_len and data[-index_len] == last:
            index = data[-index_len + 1 : -1]
            sizes = [
                int.from_bytes(index[i * width : (i + 1) * width], "little")
                for i in range(count)
            ]
            if not any(sizes) or sum(sizes) > len(data) - index_len:
                raise _Unsupported("invalid VP9 superframe index")
            pos = 0
            for size in sizes:
                if size:
                    _check_vp9_frame_byte(data[pos])
                pos += size
            return count
    _check_vp9_frame_byte(data[0])
    return 1


def _check_vp9_frame_byte(first: int) -> None:
    # Uncompressed header: frame_marker(2) = 0b10, profile_low_bit, profile_high_bit.
    if first >> 6 != 0b10:
        raise _Unsupported("invalid VP9 frame header")
    profile = (((first >> 4) & 1) << 1) | ((first >> 5) & 1)
    if profile not in _NVDEC_VP9_PROFILES:
        raise _Unsupported(
            f"VP9 profile {profile}; NVDEC decodes only 4:2:0 profiles 0 and 2"
        )
