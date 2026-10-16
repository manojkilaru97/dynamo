# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-decode NVDEC capability check for the royalty-free codecs (VP8/VP9/AV1).

NVDEC decodes VP9 only as 4:2:0 (profiles 0 and 2) and AV1 only as Main-profile
4:2:0. PyNvVideoCodec 2.2.x cannot tell the difference: the image's libavformat
ships no codec parsers, so its demuxer reports 4:2:0 8-bit for every stream and
builds a 4:2:0 decoder. Decoding the first frame of a VP9 profile 1/3 stream
(4:2:2, 4:4:0 or 4:4:4) then segfaults the whole worker process instead of
raising, which takes down every request on it.

VP9 read from ISO-BMFF (MP4/MOV) segfaults the same way even at profile 0, so
VP9 is accepted only from Matroska/WebM, where the profile is read from
``CodecPrivate`` and the first frame header. AV1 is checked against its
configuration record (ISO-BMFF ``av1C`` / Matroska ``CodecPrivate``). Anything
not positively identified as decodable is reported unsupported: a client error
for a rare file is far better than a crashed worker.

Only the Nemotron Omni loader (``VideoLoader(nvdec_royalty_free_codecs=True)``)
imports this module; other models never route these codecs to NVDEC.
"""

from __future__ import annotations

import struct

_EBML_MAGIC = b"\x1a\x45\xdf\xa3"

_SEGMENT = 0x18538067
_TRACKS = 0x1654AE6B
_TRACK_ENTRY = 0xAE
_TRACK_NUMBER = 0xD7
_CODEC_ID = 0x86
_CODEC_PRIVATE = 0x63A2
_CLUSTER = 0x1F43B675
_BLOCK_GROUP = 0xA0
_BLOCK = 0xA1
_SIMPLE_BLOCK = 0xA3
_MASTERS = frozenset({_SEGMENT, _TRACKS, _TRACK_ENTRY, _CLUSTER, _BLOCK_GROUP})
# Segment children. Inside an unknown-size master (live WebM, e.g. browser
# MediaRecorder output) one of these marks the end of that master.
_LEVEL1 = frozenset(
    {
        0x114D9B74,  # SeekHead
        0x1549A966,  # Info
        _TRACKS,
        0x1C53BB6B,  # Cues
        _CLUSTER,
        0x1254C367,  # Tags
        0x1043A770,  # Chapters
        0x1941A469,  # Attachments
    }
)
_MAX_EBML_ELEMENTS = 100_000
_MAX_EBML_DEPTH = 4

_MKV_CODEC_IDS = {"vp9": "V_VP9", "av1": "V_AV1"}
_NVDEC_VP9_PROFILES = frozenset({0, 2})
# vpcC / Matroska VP9 feature chroma values 0 and 1 are both 4:2:0.
_VP9_CHROMA_420 = frozenset({0, 1})
_MAX_CONFIG_BOX = 1024


def unsupported_reason(codec: str | None, data: bytes) -> str | None:
    """Why NVDEC must not decode ``data``, or ``None`` if it can.

    ``codec`` is the routing hint from ``probe_video_codec``. VP8 has a single
    4:2:0 8-bit profile, so it is always decodable.
    """
    if codec == "vp8":
        return None
    if codec not in _MKV_CODEC_IDS:
        return f"codec {codec!r} is not a royalty-free NVDEC codec"
    try:
        if data[:4] == _EBML_MAGIC:
            return _matroska_reason(codec, data)
        return _isobmff_reason(codec, data)
    except Exception as exc:  # noqa: BLE001 - malformed input must never escape
        return f"could not parse the {codec} stream header ({type(exc).__name__})"


def _vp9_reason(profile: int, chroma: int | None, source: str) -> str | None:
    if profile not in _NVDEC_VP9_PROFILES:
        return f"VP9 profile {profile} ({source}); NVDEC decodes only 4:2:0 profiles 0 and 2"
    if chroma is not None and chroma not in _VP9_CHROMA_420:
        return f"VP9 chroma subsampling {chroma} ({source}); NVDEC decodes only 4:2:0"
    return None


def _av1_config_reason(record: bytes, source: str) -> str | None:
    """Check an AV1CodecConfigurationRecord (ISO-BMFF av1C / Matroska CodecPrivate)."""
    if len(record) < 4 or record[0] != 0x81:
        return f"missing or invalid AV1 configuration record ({source})"
    profile = record[1] >> 5
    flags = record[2]
    twelve_bit = (flags >> 5) & 1
    monochrome = (flags >> 4) & 1
    subsampling = ((flags >> 3) & 1, (flags >> 2) & 1)
    if profile != 0 or twelve_bit or monochrome or subsampling != (1, 1):
        return (
            f"AV1 profile {profile} (12-bit={twelve_bit}, monochrome={monochrome}, "
            f"subsampling={subsampling}; {source}); NVDEC decodes only Main 4:2:0"
        )
    return None


def _config_boxes(data: bytes, fourcc: bytes) -> list[bytes]:
    """Payloads of every well-formed ``fourcc`` box, wherever it sits.

    Requiring a plausible box size keeps a stray fourcc inside media data from
    being taken for a configuration box.
    """
    found = []
    i = data.find(fourcc, 4)
    while i != -1:
        size = struct.unpack_from(">I", data, i - 4)[0]
        if 12 <= size <= _MAX_CONFIG_BOX and i - 4 + size <= len(data):
            found.append(data[i + 4 : i - 4 + size])
        i = data.find(fourcc, i + 1)
    return found


def _isobmff_reason(codec: str, data: bytes) -> str | None:
    if codec == "vp9":
        # PyNvVideoCodec 2.2.x segfaults on VP9 read from ISO-BMFF even for
        # profile 0 (measured on H100); only Matroska/WebM VP9 is safe.
        return "VP9 outside Matroska/WebM; PyNvVideoCodec cannot decode it safely"
    boxes = _config_boxes(data, b"av1C")
    if not boxes:
        return "AV1 in ISO-BMFF without an av1C box"
    for box in boxes:
        reason = _av1_config_reason(box, "av1C")
        if reason:
            return reason
    return None


def _read_vint(data: bytes, pos: int, end: int, strip: bool) -> tuple[int, int, bool]:
    """Read an EBML variable-length integer -> (value, next_pos, is_unknown_size)."""
    if pos >= end:
        raise ValueError("truncated EBML element")
    first = data[pos]
    if first == 0:
        raise ValueError("invalid EBML vint")
    length = 9 - first.bit_length()
    if pos + length > end:
        raise ValueError("truncated EBML vint")
    value = first & (0xFF >> length) if strip else first
    for b in data[pos + 1 : pos + length]:
        value = (value << 8) | b
    unknown = strip and value == (1 << (7 * length)) - 1
    return value, pos + length, unknown


class _MkvState:
    def __init__(self, codec_id: str) -> None:
        self.codec_id = codec_id
        self.steps = 0
        self.done = False
        self.entry: dict | None = None
        self.tracks: dict[int, dict] = {}
        # First byte of the first unlaced block's frame, per track number.
        self.first_frame_byte: dict[int, int] = {}

    def wanted_tracks(self) -> list[int]:
        return [n for n, t in self.tracks.items() if t.get("codec_id") == self.codec_id]


def _walk_ebml(
    data: bytes, start: int, end: int, depth: int, parent: int | None, st: _MkvState
) -> int:
    """Walk EBML children of ``[start, end)``; return where the walk stopped.

    ``parent`` is set only for an unknown-size master, whose extent is implied:
    a level-1 element ends a level-1 master such as a live-WebM Cluster.
    """
    pos = start
    while pos < end and not st.done:
        st.steps += 1
        if st.steps > _MAX_EBML_ELEMENTS:
            raise ValueError("too many EBML elements")
        eid, body, _ = _read_vint(data, pos, end, strip=False)
        if parent in _LEVEL1 and eid in _LEVEL1:
            return pos
        size, body, unknown = _read_vint(data, body, end, strip=True)
        child_end = end if unknown else min(body + size, end)
        if eid in _MASTERS and depth < _MAX_EBML_DEPTH:
            if eid == _TRACK_ENTRY:
                st.entry = {}
            try:
                stop = _walk_ebml(
                    data, body, child_end, depth + 1, eid if unknown else None, st
                )
            finally:
                if eid == _TRACK_ENTRY and st.entry is not None:
                    if "number" in st.entry:
                        st.tracks.setdefault(st.entry["number"], st.entry)
                    st.entry = None
            pos = stop if unknown else child_end
            continue
        if st.entry is not None and eid == _TRACK_NUMBER:
            st.entry["number"] = int.from_bytes(data[body:child_end], "big")
        elif st.entry is not None and eid == _CODEC_ID:
            st.entry["codec_id"] = (
                data[body:child_end].rstrip(b"\x00").decode("ascii", "replace")
            )
        elif st.entry is not None and eid == _CODEC_PRIVATE:
            st.entry["private"] = data[body:child_end]
        elif eid in (_SIMPLE_BLOCK, _BLOCK) and body < child_end:
            # Block header: track number (vint), timecode (2), flags (1), frame.
            track, hdr, _ = _read_vint(data, body, child_end, strip=True)
            # A laced block's first frame does not start the payload; skip it.
            laced = hdr + 2 < child_end and (data[hdr + 2] >> 1) & 0x3
            if track not in st.first_frame_byte and not laced and hdr + 3 < child_end:
                st.first_frame_byte[track] = data[hdr + 3]
                wanted = st.wanted_tracks()
                st.done = bool(wanted) and all(n in st.first_frame_byte for n in wanted)
        pos = child_end
    return pos


def _vp9_codec_private_reason(private: bytes) -> str | None:
    """Matroska VP9 CodecPrivate: (id, length, value) feature records."""
    profile = chroma = None
    i = 0
    while i + 2 <= len(private):
        fid, flen = private[i], private[i + 1]
        value = private[i + 2 : i + 2 + flen]
        if len(value) != flen:
            return "invalid VP9 CodecPrivate"
        if fid == 1 and flen == 1:
            profile = value[0]
        elif fid == 4 and flen == 1:
            chroma = value[0]
        i += 2 + flen
    if profile is None:
        return None
    return _vp9_reason(profile, chroma, "CodecPrivate")


def _matroska_reason(codec: str, data: bytes) -> str | None:
    want = _MKV_CODEC_IDS[codec]
    st = _MkvState(want)
    try:
        _walk_ebml(data, 0, len(data), 0, None, st)
    except (ValueError, IndexError):
        pass  # truncated or damaged tail: judge what was read before it
    numbers = st.wanted_tracks()
    if not numbers:
        return f"no {want} track in the Matroska header"
    for number in numbers:
        track = st.tracks[number]
        private = track.get("private")
        if codec == "av1":
            reason = _av1_config_reason(private or b"", "CodecPrivate")
            if reason:
                return reason
            continue
        if private:
            reason = _vp9_codec_private_reason(private)
            if reason:
                return reason
        # The uncompressed header of the first frame is authoritative:
        # frame_marker(2) = 0b10, profile_low_bit(1), profile_high_bit(1).
        first = st.first_frame_byte.get(number)
        if first is None:
            return "no unlaced VP9 frame found to read the profile from"
        if first >> 6 != 0b10:
            return "invalid VP9 frame header"
        profile = (((first >> 4) & 1) << 1) | ((first >> 5) & 1)
        reason = _vp9_reason(profile, None, "frame header")
        if reason:
            return reason
    return None
