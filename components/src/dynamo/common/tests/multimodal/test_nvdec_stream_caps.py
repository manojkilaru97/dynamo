# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Omni pre-NVDEC safety check (synthetic containers)."""

import struct
import time

import pytest

from dynamo.common.multimodal.nvdec_decoder import probe_video_codec
from dynamo.common.multimodal.nvdec_stream_caps import unsupported_reason

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]

# First byte of a VP9 keyframe as written by libvpx, per profile (not derived
# from the parser's bit layout, so a swapped profile bit would be caught).
VP9_KEYFRAME = {0: 0x82, 1: 0xA2, 2: 0x92, 3: 0xB1}
VP9_INTER = {0: 0x86, 1: 0xA6}
UNKNOWN = b"\x01\xff\xff\xff\xff\xff\xff\xff"


# --- Matroska builders -------------------------------------------------------------


def _el(eid: int, payload: bytes, unknown: bool = False) -> bytes:
    ident = eid.to_bytes((eid.bit_length() + 7) // 8, "big")
    if unknown:
        return ident + UNKNOWN + payload
    return ident + (0x10000000 | len(payload)).to_bytes(4, "big") + payload


def _block(track: int, frame: bytes, laced: bool = False) -> bytes:
    flags = 0x80 | (0x02 if laced else 0)
    return _el(0xA3, bytes([0x80 | track]) + b"\x00\x00" + bytes([flags]) + frame)


def _frame(first: int, size: int = 12) -> bytes:
    return bytes([first]) + b"\x10" * (size - 1)


def _superframe(*frames: bytes) -> bytes:
    marker = 0xC0 | (0 << 3) | (len(frames) - 1)  # 1-byte sizes
    return b"".join(frames) + bytes([marker, *[len(f) for f in frames], marker])


def _track(
    number: int,
    codec_id: str,
    private: bytes | None = None,
    extra: bytes = b"",
    ttype: int | None = 1,
) -> bytes:
    body = _el(0xD7, bytes([number])) + _el(0x86, codec_id.encode())
    if ttype is not None:
        body += _el(0x83, bytes([ttype]))
    if private is not None:
        body += _el(0x63A2, private)
    return _el(0xAE, body + extra)


def _mkv(
    tracks: list[bytes], clusters: list[list[bytes]], unknown: bool = False
) -> bytes:
    body = _el(0x1654AE6B, b"".join(tracks))
    for blocks in clusters:
        body += _el(0x1F43B675, _el(0xE7, b"\x00") + b"".join(blocks), unknown=unknown)
    return _el(0x1A45DFA3, _el(0x4282, b"webm")) + _el(
        0x18538067, body, unknown=unknown
    )


def _vp9_mkv(*firsts: int, **kw) -> bytes:
    tracks = [_track(1, "V_VP9"), _track(2, "A_OPUS", ttype=2)]
    return _mkv(tracks, [[_block(1, _frame(f)) for f in firsts]], **kw)


# --- ISO-BMFF builders -------------------------------------------------------------


def _box(fourcc: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + fourcc + payload


def _full(fourcc: bytes, payload: bytes) -> bytes:
    return _box(fourcc, b"\x00\x00\x00\x00" + payload)


def _mp4(
    entry_fourcc: bytes,
    entry_children: bytes = b"",
    mdat: bytes = b"\x00" * 64,
    handler: bytes = b"vide",
) -> bytes:
    entry = _box(entry_fourcc, b"\x00" * 78 + entry_children)
    stsd = _full(b"stsd", struct.pack(">I", 1) + entry)
    hdlr = _full(b"hdlr", b"\x00\x00\x00\x00" + handler + b"\x00" * 13)
    mdia = _box(b"mdia", hdlr + _box(b"minf", _box(b"stbl", stsd)))
    moov = _box(b"moov", _box(b"trak", mdia))
    return _box(b"ftyp", b"isom\x00\x00\x02\x00") + moov + _box(b"mdat", mdat)


def _av1c(
    profile: int = 0, mono: int = 0, ssx: int = 1, ssy: int = 1, twelve: int = 0
) -> bytes:
    flags = twelve << 5 | mono << 4 | ssx << 3 | ssy << 2
    return bytes([0x81, profile << 5 | 8, flags, 0])


def _rf(data: bytes) -> str | None:
    return unsupported_reason(data, strict=True)


# --- VP9 in Matroska -----------------------------------------------------------------


@pytest.mark.parametrize("profile", [0, 2])
def test_vp9_420_profiles_are_decodable(profile):
    assert (
        _rf(_vp9_mkv(VP9_KEYFRAME[profile], VP9_INTER[0] if profile == 0 else 0x96))
        is None
    )


@pytest.mark.parametrize("profile", [1, 3])
def test_vp9_non_420_profiles_are_rejected(profile):
    reason = _rf(_vp9_mkv(VP9_KEYFRAME[profile]))
    assert reason is not None and f"VP9 profile {profile}" in reason


def test_vp9_live_webm_with_unknown_sizes_is_parsed():
    """Browser MediaRecorder WebM uses unknown-size Segment and Cluster."""
    assert _rf(_vp9_mkv(VP9_KEYFRAME[0], VP9_INTER[0], unknown=True)) is None
    assert _rf(_vp9_mkv(VP9_KEYFRAME[0], VP9_KEYFRAME[1], unknown=True)) is not None


def test_vp9_every_frame_is_checked():
    """VP9 carries the profile per frame: a later profile-1 keyframe is rejected."""
    assert _rf(_vp9_mkv(VP9_KEYFRAME[0], VP9_INTER[0], VP9_KEYFRAME[1])) is not None
    later_cluster = _mkv(
        [_track(1, "V_VP9")],
        [[_block(1, _frame(VP9_KEYFRAME[0]))], [_block(1, _frame(VP9_KEYFRAME[1]))]],
    )
    assert _rf(later_cluster) is not None


def test_vp9_superframe_frames_are_each_checked():
    hidden_p1 = _superframe(_frame(VP9_KEYFRAME[0]), _frame(VP9_INTER[1]))
    data = _mkv([_track(1, "V_VP9")], [[_block(1, hidden_p1)]])
    assert _rf(data) is not None
    ok = _superframe(_frame(VP9_KEYFRAME[0]), _frame(VP9_INTER[0]))
    assert _rf(_mkv([_track(1, "V_VP9")], [[_block(1, ok)]])) is None


def test_vp9_index_lookalike_that_does_not_fit_is_a_single_frame():
    frame = _frame(VP9_KEYFRAME[0], 6) + bytes([0xC1, 0xFF, 0xFF, 0xC1])
    assert _rf(_mkv([_track(1, "V_VP9")], [[_block(1, frame)]])) is None


def test_vp9_laced_blocks_are_rejected():
    data = _mkv(
        [_track(1, "V_VP9")], [[_block(1, _frame(VP9_KEYFRAME[0]), laced=True)]]
    )
    assert "laced" in _rf(data)


def test_vp9_content_encoded_track_is_rejected():
    """Header stripping could hide the profile byte the gate reads."""
    encodings = _el(
        0x6D80,
        _el(0x6240, _el(0x5034, _el(0x4254, b"\x03") + _el(0x4255, b"\xa2\x49"))),
    )
    data = _mkv(
        [_track(1, "V_VP9", extra=encodings)],
        [[_block(1, bytes([0x83]) + b"\x00" * 11)]],
    )
    assert "content encoding" in _rf(data)


def test_vp9_codec_private_is_checked_too():
    bad_chroma = bytes([1, 1, 0, 4, 1, 3])
    data = _mkv(
        [_track(1, "V_VP9", private=bad_chroma)], [[_block(1, _frame(VP9_KEYFRAME[0]))]]
    )
    assert "chroma" in _rf(data)
    good = bytes([1, 1, 0, 3, 1, 8, 4, 1, 1])
    data = _mkv(
        [_track(1, "V_VP9", private=good)], [[_block(1, _frame(VP9_KEYFRAME[0]))]]
    )
    assert _rf(data) is None
    data = _mkv(
        [_track(1, "V_VP9", private=bytes([1, 1, 0]))],
        [[_block(1, _frame(VP9_KEYFRAME[1]))]],
    )
    assert _rf(data) is not None  # the frame header still decides


def test_vp9_blocks_of_other_tracks_are_ignored():
    tracks = [_track(1, "V_VP9"), _track(2, "A_OPUS", ttype=2)]
    data = _mkv(tracks, [[_block(2, b"\xff" * 12), _block(1, _frame(VP9_KEYFRAME[0]))]])
    assert _rf(data) is None


def test_vp9_truncated_upload_keeps_what_was_read():
    data = _vp9_mkv(VP9_KEYFRAME[0], VP9_INTER[0], VP9_INTER[0])
    assert _rf(data[:-5]) is None
    data = _vp9_mkv(VP9_KEYFRAME[1], VP9_INTER[0], VP9_INTER[0])
    assert _rf(data[:-5]) is not None


def test_vp9_damage_before_the_end_is_rejected():
    """A demuxer may resync past damage; the frames after it were never inspected."""
    good = _block(1, _frame(VP9_KEYFRAME[0]))
    # A SimpleBlock whose size overruns its Cluster, followed by another Cluster.
    oversized = (
        b"\xa3" + (0x10000000 | 4096).to_bytes(4, "big") + b"\x81\x00\x00\x80\x82"
    )
    data = _mkv([_track(1, "V_VP9")], [[good, oversized], [good]])
    assert "could not be fully inspected" in _rf(data)


def test_duplicate_track_numbers_are_rejected():
    data = _mkv(
        [_track(1, "V_MPEG4/ISO/AVC"), _track(1, "V_VP9")],
        [[_block(1, _frame(VP9_KEYFRAME[1]))]],
    )
    assert "duplicate" in unsupported_reason(data, strict=False)


@pytest.mark.parametrize(
    "data",
    [
        _mkv([_track(1, "V_VP9")], []),  # no frame
        _vp9_mkv(0x00),  # bad frame marker
        b"\x1a\x45\xdf\xa3",  # magic only
        b"\x1a\x45\xdf\xa3\x00\x00\x00",  # invalid vint
    ],
)
def test_unidentifiable_vp9_streams_are_rejected(data):
    assert _rf(data) is not None


def test_garbage_never_raises():
    base = _vp9_mkv(VP9_KEYFRAME[0], VP9_INTER[0])
    for i in range(len(base)):
        for flip in (0x01, 0x80, 0xFF):
            mutated = bytearray(base)
            mutated[i] ^= flip
            for strict in (True, False):
                result = unsupported_reason(bytes(mutated), strict=strict)
                assert result is None or isinstance(result, str)


# --- VP8 -------------------------------------------------------------------------------


def test_vp8_is_decodable_only_from_matroska():
    assert _rf(_mkv([_track(1, "V_VP8")], [[_block(1, b"\x10" * 12)]])) is None
    assert "outside Matroska" in _rf(_mp4(b"vp08"))


# --- ISO-BMFF ----------------------------------------------------------------------------


def test_mp4_vp9_is_always_rejected():
    """PyNvVideoCodec segfaults on VP9 read from ISO-BMFF even at profile 0."""
    vpcc = _full(b"vpcC", bytes([0, 31, 0x80, 1, 1, 1, 0, 0]))
    assert "outside Matroska" in _rf(_mp4(b"vp09", vpcc))
    assert "outside Matroska" in unsupported_reason(_mp4(b"vp09", vpcc), strict=False)


def test_mp4_non_video_tracks_are_ignored():
    assert _rf(_mp4(b"vp09", handler=b"soun")) == "no video track found"


def test_av1_main_420_is_decodable():
    assert _rf(_mp4(b"av01", _box(b"av1C", _av1c()))) is None
    assert (
        _rf(_mkv([_track(1, "V_AV1", private=_av1c())], [[_block(1, b"\x12" * 12)]]))
        is None
    )


@pytest.mark.parametrize(
    "record",
    [
        _av1c(profile=1, ssx=0, ssy=0),
        _av1c(profile=2),
        _av1c(mono=1),
        _av1c(ssy=0),
        _av1c(twelve=1),
        b"\x00\x00\x00\x00",
    ],
)
def test_av1_other_profiles_are_rejected(record):
    assert _rf(_mp4(b"av01", _box(b"av1C", record))) is not None
    assert (
        _rf(_mkv([_track(1, "V_AV1", private=record)], [[_block(1, b"\x12" * 12)]]))
        is not None
    )


def test_av1_without_config_record_is_rejected():
    assert _rf(_mp4(b"av01")) is not None
    assert _rf(_mkv([_track(1, "V_AV1")], [[_block(1, b"\x12" * 12)]])) is not None


def test_av1_lookalike_box_outside_the_sample_entry_is_ignored():
    """Only the real av01 sample entry's av1C counts, not one inside metadata."""
    fake = _box(b"av1C", _av1c())
    data = _mp4(b"av01", _box(b"av1C", _av1c(profile=2)), mdat=fake * 4)
    assert _rf(data) is not None


# --- Misprobes: the byte-marker hint disagrees with the real track -------------------------


def test_h264_with_a_stray_av1_marker_keeps_nvdec():
    data = _mp4(b"avc1", mdat=b"\x00" * 16 + b"av01" + b"\x00" * 16)
    assert probe_video_codec(data) == "av1"
    assert unsupported_reason(data, strict=True) is None
    mkv = _mkv([_track(1, "V_MPEG4/ISO/AVC")], [[_block(1, b"V_AV1" + b"\x00" * 8)]])
    assert probe_video_codec(mkv) == "av1"
    assert unsupported_reason(mkv, strict=True) is None


def test_vp9_profile1_misprobed_as_h264_is_still_rejected():
    frame = bytes([VP9_KEYFRAME[1]]) + b"avc1" + b"\x00" * 8
    data = _mkv([_track(1, "V_VP9")], [[_block(1, frame)]])
    assert probe_video_codec(data) == "h264"
    assert unsupported_reason(data, strict=False) is not None


def test_non_strict_keeps_unidentified_containers():
    """An H.264/HEVC hint is only refused on a positively unsafe track."""
    for data in (b"\x00\x00\x01\xb3" + b"\x00" * 64, b"RIFF" + b"\x00" * 64, b""):
        assert unsupported_reason(data, strict=False) is None
        assert unsupported_reason(data, strict=True) is not None


def test_strict_rejects_unrecognised_codecs():
    assert "unrecognised" in _rf(_mp4(b"mp4v"))
    assert unsupported_reason(_mp4(b"mp4v"), strict=False) is None


# --- Bounded work ------------------------------------------------------------------------


def _timed(data: bytes, strict: bool = True) -> tuple[str | None, float]:
    t = time.perf_counter()
    reason = unsupported_reason(data, strict=strict)
    return reason, time.perf_counter() - t


def test_many_tracks_are_rejected_quickly():
    tracks = [_track((i % 120) + 1, "V_VP9") for i in range(16_000)]
    reason, elapsed = _timed(_mkv(tracks, []))
    assert reason is not None and elapsed < 1.0


def test_blocks_for_unknown_tracks_cost_constant_time():
    blocks = [_block(3 + (i % 100), b"\x00" * 4) for i in range(50_000)]
    data = _mkv([_track(1, "V_VP9")], [blocks + [_block(1, _frame(VP9_KEYFRAME[0]))]])
    reason, elapsed = _timed(data)
    assert reason is None and elapsed < 2.0


def test_repeated_lookalike_boxes_do_not_amplify():
    junk = b"\x00\x00\x04\x00av1C" * (2 << 20 >> 3)
    reason, elapsed = _timed(_mp4(b"av01", _box(b"av1C", _av1c()), mdat=junk))
    assert reason is None and elapsed < 1.0


def test_large_h264_mp4_is_checked_without_scanning_media_data():
    reason, elapsed = _timed(_mp4(b"avc1", mdat=b"\x00" * (64 << 20)), strict=False)
    assert reason is None and elapsed < 0.5
