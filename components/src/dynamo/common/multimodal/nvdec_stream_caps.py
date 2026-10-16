# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-decode NVDEC safety check for the Nemotron Omni video path.

PyNvVideoCodec 2.2.x segfaults the whole worker process, instead of raising,
on the first decoded frame of:

* VP9 profile 1 or 3 (4:2:2, 4:4:0, 4:4:4), in any container. The image's
  libavformat ships no codec parsers, so the demuxer reports 4:2:0 8-bit for
  every stream and builds a 4:2:0 decoder.
* VP9 read from ISO-BMFF (MP4/MOV), even at profile 0.

Omni routes VP8/VP9/AV1 to NVDEC, so a single such upload took down a worker,
and request migration plus client retries carried it across the fleet.

This module identifies the real video tracks from the container structure
(ISO-BMFF ``moov``/``stsd`` sample entries, Matroska ``Tracks``) rather than
from byte markers, then:

* VP8: Matroska only.
* VP9: Matroska only; ``CodecPrivate`` (when present) and every frame header,
  including each frame of a superframe, must be profile 0 or 2 (4:2:0).
  Laced or content-encoded (e.g. header-stripped) tracks are rejected.
* AV1: the configuration record must be Main profile, 8/10-bit, 4:2:0.
* H.264/HEVC tracks are left alone: their NVDEC routing is unchanged.

With ``strict`` (a VP8/VP9/AV1 routing hint), anything that cannot be
positively identified as safe is rejected: a client error for a rare file is
far better than a crashed worker. Without it (an H.264/HEVC hint), only a
positively identified unsafe track is rejected, so H.264/HEVC behave as before.

Work is bounded (box/element/track budgets, O(1) per Matroska block, no data
copies beyond small header fields). Only the Omni loader imports this module.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

_ROYALTY_FREE = frozenset({"vp8", "vp9", "av1"})
_NVDEC_VP9_PROFILES = frozenset({0, 2})
# Matroska VP9 CodecPrivate chroma values 0 and 1 are both 4:2:0.
_VP9_CHROMA_420 = frozenset({0, 1})

_MAX_TRACKS = 64
_MAX_BOXES = 10_000
_MAX_EBML_ELEMENTS = 2_000_000
_MAX_EBML_DEPTH = 5
_MAX_CONFIG_BYTES = 64 * 1024


class _Unsupported(Exception):
    """A positively identified condition NVDEC must not see."""


class _Unparseable(Exception):
    """The container could not be identified or parsed."""


class _Truncated(_Unparseable):
    """The data ends inside an element (a cut-off upload)."""


@dataclass
class _Track:
    codec: str  # vp8 | vp9 | av1 | h264 | hevc | other:<id>
    container: str  # isobmff | matroska
    config: bytes | None = None  # av1C / Matroska CodecPrivate
    encoded: bool = False  # Matroska ContentEncodings present
    number: int | None = None
    vp9_frames: int = 0


def unsupported_reason(data: bytes, strict: bool) -> str | None:
    """Why NVDEC must not decode ``data``, or ``None`` if it may.

    ``strict``: fail closed when the stream cannot be positively identified
    as safe. Use it when the routing hint is VP8/VP9/AV1.
    """
    try:
        tracks = _video_tracks(data)
    except _Unsupported as exc:
        return str(exc)
    except _Unparseable as exc:
        return str(exc) if strict else None
    except Exception as exc:  # noqa: BLE001 - malformed input must never escape
        return (
            f"could not parse the video container ({type(exc).__name__})"
            if strict
            else None
        )
    if not tracks:
        return "no video track found" if strict else None
    for track in tracks:
        if track.codec in _ROYALTY_FREE:
            reason = _royalty_free_reason(track)
            if reason:
                return reason
        elif track.codec not in ("h264", "hevc") and strict:
            return f"unrecognised video codec {track.codec}"
    return None


def _royalty_free_reason(track: _Track) -> str | None:
    if track.container != "matroska" and track.codec in ("vp8", "vp9"):
        return f"{track.codec.upper()} outside Matroska/WebM; PyNvVideoCodec cannot decode it safely"
    if track.encoded:
        return f"{track.codec.upper()} track uses Matroska content encoding"
    if track.codec == "av1":
        return _av1_config_reason(track.config or b"")
    if track.codec == "vp9":
        if track.config:
            reason = _vp9_codec_private_reason(track.config)
            if reason:
                return reason
        if track.vp9_frames == 0:
            return "no VP9 frame found to read the profile from"
    return None


def _vp9_reason(profile: int, chroma: int | None, source: str) -> str | None:
    if profile not in _NVDEC_VP9_PROFILES:
        return f"VP9 profile {profile} ({source}); NVDEC decodes only 4:2:0 profiles 0 and 2"
    if chroma is not None and chroma not in _VP9_CHROMA_420:
        return f"VP9 chroma subsampling {chroma} ({source}); NVDEC decodes only 4:2:0"
    return None


def _av1_config_reason(record: bytes) -> str | None:
    """Check an AV1CodecConfigurationRecord (ISO-BMFF av1C / Matroska CodecPrivate)."""
    if len(record) < 4 or record[0] != 0x81:
        return "missing or invalid AV1 configuration record"
    profile = record[1] >> 5
    flags = record[2]
    twelve_bit = (flags >> 5) & 1
    monochrome = (flags >> 4) & 1
    subsampling = ((flags >> 3) & 1, (flags >> 2) & 1)
    if profile != 0 or twelve_bit or monochrome or subsampling != (1, 1):
        return (
            f"AV1 profile {profile} (12-bit={twelve_bit}, monochrome={monochrome}, "
            f"subsampling={subsampling}); NVDEC decodes only Main 4:2:0"
        )
    return None


def _vp9_codec_private_reason(private: bytes) -> str | None:
    """Matroska VP9 CodecPrivate: (id, length, value) feature records."""
    profile = chroma = None
    i = 0
    while i < len(private):
        if i + 2 > len(private):
            return "invalid VP9 CodecPrivate"
        fid, flen = private[i], private[i + 1]
        if i + 2 + flen > len(private):
            return "invalid VP9 CodecPrivate"
        if fid == 1 and flen == 1:
            profile = private[i + 2]
        elif fid == 4 and flen == 1:
            chroma = private[i + 2]
        i += 2 + flen
    if profile is None:
        return None
    return _vp9_reason(profile, chroma, "CodecPrivate")


def _check_vp9_frame_byte(first: int) -> None:
    # Uncompressed header: frame_marker(2) = 0b10, profile_low_bit, profile_high_bit.
    if first >> 6 != 0b10:
        raise _Unsupported("invalid VP9 frame header")
    profile = (((first >> 4) & 1) << 1) | ((first >> 5) & 1)
    reason = _vp9_reason(profile, None, "frame header")
    if reason:
        raise _Unsupported(reason)


def _check_vp9_block(data: bytes, start: int, end: int) -> int:
    """Check every frame header in one VP9 block payload; return the frame count.

    A superframe ends with an index whose first and last bytes are the same
    marker 0b110xxxxx (VP9 spec, Annex B). An index whose sizes do not fit is
    not a superframe; decoders then read the payload as a single frame.
    """
    if start >= end:
        raise _Unsupported("empty VP9 frame")
    last = data[end - 1]
    if last & 0xE0 == 0xC0:
        count = (last & 0x7) + 1
        size_bytes = ((last >> 3) & 0x3) + 1
        index_len = 2 + size_bytes * count
        if end - start >= index_len and data[end - index_len] == last:
            entry = end - index_len + 1
            sizes = [
                int.from_bytes(
                    data[entry + i * size_bytes : entry + (i + 1) * size_bytes],
                    "little",
                )
                for i in range(count)
            ]
            if all(sizes) and sum(sizes) <= end - index_len - start:
                pos = start
                for size in sizes:
                    _check_vp9_frame_byte(data[pos])
                    pos += size
                return count
    _check_vp9_frame_byte(data[start])
    return 1


def _video_tracks(data: bytes) -> list[_Track]:
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return _matroska_tracks(data)
    if data[4:8] in (
        b"ftyp",
        b"moov",
        b"free",
        b"skip",
        b"mdat",
        b"wide",
        b"styp",
        b"pdin",
    ):
        return _isobmff_tracks(data)
    raise _Unparseable("unrecognised video container")


# --- ISO-BMFF ------------------------------------------------------------------

_ISO_CODECS = {
    b"avc1": "h264",
    b"avc2": "h264",
    b"avc3": "h264",
    b"avc4": "h264",
    b"hvc1": "hevc",
    b"hev1": "hevc",
    b"vp08": "vp8",
    b"vp09": "vp9",
    b"av01": "av1",
}
# VisualSampleEntry fields before its child boxes (ISO/IEC 14496-12 §12.1.3).
_VISUAL_SAMPLE_ENTRY_FIELDS = 78


class _BoxBudget:
    def __init__(self) -> None:
        self.left = _MAX_BOXES


def _boxes(data: bytes, start: int, end: int, budget: _BoxBudget):
    pos = start
    while pos + 8 <= end:
        budget.left -= 1
        if budget.left < 0:
            raise _Unparseable("too many ISO-BMFF boxes")
        size, kind = struct.unpack_from(">I4s", data, pos)
        header = 8
        if size == 1:
            if pos + 16 > end:
                raise _Unparseable("truncated ISO-BMFF box")
            size = struct.unpack_from(">Q", data, pos + 8)[0]
            header = 16
        elif size == 0:
            size = end - pos
        if size < header:
            raise _Unparseable("invalid ISO-BMFF box size")
        stop = min(pos + size, end)  # a truncated final box keeps what it has
        yield kind, pos + header, stop
        pos += size


def _child(data: bytes, start: int, end: int, kind: bytes, budget: _BoxBudget):
    for k, s, e in _boxes(data, start, end, budget):
        if k == kind:
            return s, e
    return None


def _isobmff_tracks(data: bytes) -> list[_Track]:
    budget = _BoxBudget()
    moov = _child(data, 0, len(data), b"moov", budget)
    if moov is None:
        raise _Unparseable("no moov box")
    tracks: list[_Track] = []
    for kind, ts, te in _boxes(data, *moov, budget):
        if kind != b"trak":
            continue
        mdia = _child(data, ts, te, b"mdia", budget)
        if mdia is None:
            continue
        hdlr = _child(data, *mdia, b"hdlr", budget)
        # hdlr: FullBox (4) + pre_defined (4) + handler_type (4)
        if (
            hdlr is None
            or hdlr[1] - hdlr[0] < 12
            or data[hdlr[0] + 8 : hdlr[0] + 12] != b"vide"
        ):
            continue
        minf = _child(data, *mdia, b"minf", budget)
        stbl = minf and _child(data, *minf, b"stbl", budget)
        stsd = stbl and _child(data, *stbl, b"stsd", budget)
        if not stsd:
            raise _Unparseable("video track without a sample description")
        # stsd: FullBox (4) + entry_count (4), then sample entries.
        for fourcc, es, ee in _boxes(data, stsd[0] + 8, stsd[1], budget):
            codec = _ISO_CODECS.get(fourcc, "other:" + fourcc.decode("latin-1"))
            track = _Track(codec=codec, container="isobmff")
            if codec == "av1":
                cfg = _child(
                    data, es + _VISUAL_SAMPLE_ENTRY_FIELDS, ee, b"av1C", budget
                )
                track.config = (
                    data[cfg[0] : min(cfg[1], cfg[0] + _MAX_CONFIG_BYTES)]
                    if cfg
                    else None
                )
            tracks.append(track)
            if len(tracks) > _MAX_TRACKS:
                raise _Unparseable("too many video tracks")
    return tracks


# --- Matroska / WebM -----------------------------------------------------------

_SEGMENT = 0x18538067
_TRACKS = 0x1654AE6B
_TRACK_ENTRY = 0xAE
_TRACK_NUMBER = 0xD7
_TRACK_TYPE = 0x83
_CODEC_ID = 0x86
_CODEC_PRIVATE = 0x63A2
_CONTENT_ENCODINGS = 0x6D80
_CLUSTER = 0x1F43B675
_BLOCK_GROUP = 0xA0
_BLOCK = 0xA1
_SIMPLE_BLOCK = 0xA3
_MASTERS = frozenset({_SEGMENT, _TRACKS, _TRACK_ENTRY, _CLUSTER, _BLOCK_GROUP})
# Segment children. Inside an unknown-size master (live WebM, e.g. browser
# MediaRecorder output) one of these ends a level-1 master such as a Cluster.
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
_MKV_CODECS = {
    "V_VP8": "vp8",
    "V_VP9": "vp9",
    "V_AV1": "av1",
    "V_MPEG4/ISO/AVC": "h264",
    "V_MPEGH/ISO/HEVC": "hevc",
}


def _read_vint(data: bytes, pos: int, end: int, strip: bool) -> tuple[int, int, bool]:
    """Read an EBML variable-length integer -> (value, next_pos, is_unknown_size)."""
    cut = _Truncated if end >= len(data) else _Unparseable
    if pos >= end:
        raise cut("truncated EBML element")
    first = data[pos]
    if first == 0:
        raise _Unparseable("invalid EBML vint")
    length = 9 - first.bit_length()
    if pos + length > end:
        raise cut("truncated EBML vint")
    value = first & (0xFF >> length) if strip else first
    for b in data[pos + 1 : pos + length]:
        value = (value << 8) | b
    unknown = strip and value == (1 << (7 * length)) - 1
    return value, pos + length, unknown


class _MkvState:
    def __init__(self) -> None:
        self.elements = 0
        self.done = False
        self.entry: dict | None = None
        self.entries = 0
        self.numbers: set[int] = set()
        self.tracks: dict[int, _Track] = {}
        self.vp9: dict[int, _Track] = {}  # track number -> VP9 track, O(1) per block


def _close_track_entry(st: _MkvState) -> None:
    entry, st.entry = st.entry, None
    if entry is None or "number" not in entry:
        return
    if entry["number"] in st.numbers:
        raise _Unsupported("duplicate Matroska track number")
    st.numbers.add(entry["number"])
    codec_id = entry.get("codec_id", "")
    video = entry.get("type") == 1 or (
        "type" not in entry and codec_id.startswith("V_")
    )
    if not video:
        return
    track = _Track(
        codec=_MKV_CODECS.get(codec_id, "other:" + codec_id),
        container="matroska",
        config=entry.get("private"),
        encoded=entry.get("encoded", False),
        number=entry["number"],
    )
    st.tracks[track.number] = track
    if track.codec == "vp9":
        st.vp9[track.number] = track


def _walk_ebml(
    data: bytes,
    start: int,
    end: int,
    depth: int,
    parent: int | None,
    unknown_parent: bool,
    st: _MkvState,
) -> int:
    pos = start
    while pos < end and not st.done:
        st.elements += 1
        if st.elements > _MAX_EBML_ELEMENTS:
            raise _Unparseable("Matroska stream too large to inspect")
        eid, body, _ = _read_vint(data, pos, end, strip=False)
        if unknown_parent and parent in _LEVEL1 and eid in _LEVEL1:
            return pos
        size, body, unknown = _read_vint(data, body, end, strip=True)
        if unknown:
            child_end = end
        elif body + size <= end:
            child_end = body + size
        elif end >= len(data):
            child_end = end  # a cut-off upload: keep what it has
        else:
            raise _Unparseable("Matroska element overruns its parent")
        if eid == _CLUSTER and st.tracks and not st.vp9:
            st.done = True  # tracks are known and no frame needs checking
            return pos
        if eid in _MASTERS and depth < _MAX_EBML_DEPTH:
            if eid == _TRACK_ENTRY:
                st.entries += 1
                if st.entries > _MAX_TRACKS:
                    raise _Unparseable("too many Matroska tracks")
                st.entry = {}
            try:
                stop = _walk_ebml(data, body, child_end, depth + 1, eid, unknown, st)
            finally:
                if eid == _TRACK_ENTRY:
                    _close_track_entry(st)
            pos = stop if unknown else child_end
            continue
        entry = st.entry
        if entry is not None:
            if eid in (_TRACK_NUMBER, _TRACK_TYPE):
                if child_end - body > 8:
                    raise _Unparseable("invalid Matroska track field")
                key = "number" if eid == _TRACK_NUMBER else "type"
                entry[key] = int.from_bytes(data[body:child_end], "big")
            elif eid == _CODEC_ID:
                if child_end - body > 64:
                    raise _Unparseable("invalid Matroska CodecID")
                entry["codec_id"] = (
                    data[body:child_end].rstrip(b"\x00").decode("ascii", "replace")
                )
            elif eid == _CODEC_PRIVATE:
                entry["private"] = data[body : min(child_end, body + _MAX_CONFIG_BYTES)]
            elif eid == _CONTENT_ENCODINGS:
                entry["encoded"] = True
        elif (
            eid in (_SIMPLE_BLOCK, _BLOCK)
            and parent in (_CLUSTER, _BLOCK_GROUP)
            and st.vp9
        ):
            # Block header: track number (vint), timecode (2), flags (1), frame data.
            track_no, hdr, _ = _read_vint(data, body, child_end, strip=True)
            track = st.vp9.get(track_no)
            if track is not None:
                if hdr + 3 > child_end:
                    cut = _Truncated if child_end >= len(data) else _Unparseable
                    raise cut("truncated Matroska block")
                if (data[hdr + 2] >> 1) & 0x3:
                    raise _Unsupported("laced VP9 block")
                track.vp9_frames += _check_vp9_block(data, hdr + 3, child_end)
        pos = child_end
    return pos


def _matroska_tracks(data: bytes) -> list[_Track]:
    st = _MkvState()
    try:
        _walk_ebml(data, 0, len(data), 0, None, False, st)
    except _Truncated:
        # A cut-off upload: every frame before the cut was inspected.
        if not st.tracks:
            raise
    except _Unparseable as exc:
        # Damage or an exhausted budget: a demuxer may resync past it, so a
        # VP9 track could still carry frames that were never inspected.
        if st.vp9:
            raise _Unsupported(
                f"VP9 stream could not be fully inspected ({exc})"
            ) from None
        if not st.tracks:
            raise
    return list(st.tracks.values())
