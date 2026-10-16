# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the pre-NVDEC VP9/AV1 profile check (synthetic containers)."""

import struct

import pytest

from dynamo.common.multimodal.nvdec_stream_caps import unsupported_reason

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]

UNKNOWN = b"\x01\xff\xff\xff\xff\xff\xff\xff"


def _el(eid: int, payload: bytes, unknown: bool = False) -> bytes:
    ident = eid.to_bytes((eid.bit_length() + 7) // 8, "big")
    if unknown:
        return ident + UNKNOWN + payload
    size = len(payload)
    assert size < (1 << 28)
    return ident + (0x10000000 | size).to_bytes(4, "big") + payload


def _vp9_header(profile: int) -> int:
    return 0b10 << 6 | (profile & 1) << 5 | (profile >> 1) << 4


def _simple_block(track: int, first: int, laced: bool = False) -> bytes:
    flags = 0x80 | (0x02 if laced else 0)
    return _el(
        0xA3, bytes([0x80 | track]) + b"\x00\x00" + bytes([flags, first]) + b"\x00" * 8
    )


def _webm(
    codec_id: str,
    blocks: list[bytes],
    private: bytes | None = None,
    unknown_sizes: bool = False,
    track: int = 1,
) -> bytes:
    entry = _el(0xD7, bytes([track])) + _el(0x86, codec_id.encode())
    if private is not None:
        entry += _el(0x63A2, private)
    audio = _el(0xAE, _el(0xD7, bytes([track + 1])) + _el(0x86, b"A_OPUS"))
    tracks = _el(0x1654AE6B, _el(0xAE, entry) + audio)
    cluster = _el(
        0x1F43B675, _el(0xE7, b"\x00") + b"".join(blocks), unknown=unknown_sizes
    )
    segment = _el(0x18538067, tracks + cluster, unknown=unknown_sizes)
    return _el(0x1A45DFA3, _el(0x4282, b"webm")) + segment


def _box(fourcc: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + fourcc + payload


def _vpcc(profile: int, chroma: int) -> bytes:
    payload = bytes([1, 0, 0, 0, profile, 31, (8 << 4) | (chroma << 1), 1, 1, 1, 0, 0])
    return _box(b"vpcC", payload)


def _mp4(sample_entry_child: bytes, fourcc: bytes = b"vp09") -> bytes:
    stsd = _box(b"stsd", b"\x00" * 8 + _box(fourcc, b"\x00" * 78 + sample_entry_child))
    return (
        _box(b"ftyp", b"isom\x00\x00\x02\x00")
        + _box(b"moov", _box(b"trak", stsd))
        + _box(b"mdat", b"\x00" * 64)
    )


def _av1c(
    profile: int = 0, mono: int = 0, ssx: int = 1, ssy: int = 1, twelve: int = 0
) -> bytes:
    flags = twelve << 5 | mono << 4 | ssx << 3 | ssy << 2
    return bytes([0x81, profile << 5 | 8, flags, 0])


# --- VP8 ---------------------------------------------------------------------


def test_vp8_is_always_decodable():
    assert unsupported_reason("vp8", b"anything") is None


def test_non_royalty_free_codec_is_not_cleared():
    assert unsupported_reason("h264", b"x") is not None
    assert unsupported_reason(None, b"x") is not None


# --- VP9 in Matroska/WebM ------------------------------------------------------


@pytest.mark.parametrize("profile", [0, 2])
def test_webm_vp9_420_profiles_are_decodable(profile):
    assert (
        unsupported_reason(
            "vp9", _webm("V_VP9", [_simple_block(1, _vp9_header(profile))])
        )
        is None
    )


@pytest.mark.parametrize("profile", [1, 3])
def test_webm_vp9_non_420_profiles_are_rejected(profile):
    reason = unsupported_reason(
        "vp9", _webm("V_VP9", [_simple_block(1, _vp9_header(profile))])
    )
    assert reason is not None and f"VP9 profile {profile}" in reason


def test_webm_unknown_size_live_stream_is_parsed():
    """Browser MediaRecorder WebM uses unknown-size Segment and Cluster."""
    ok = _webm("V_VP9", [_simple_block(1, _vp9_header(0))], unknown_sizes=True)
    bad = _webm("V_VP9", [_simple_block(1, _vp9_header(1))], unknown_sizes=True)
    assert unsupported_reason("vp9", ok) is None
    assert unsupported_reason("vp9", bad) is not None


def test_webm_first_video_frame_is_used_not_audio():
    blocks = [_simple_block(2, 0xFF), _simple_block(1, _vp9_header(1))]
    assert unsupported_reason("vp9", _webm("V_VP9", blocks)) is not None
    blocks = [_simple_block(2, 0x00), _simple_block(1, _vp9_header(0))]
    assert unsupported_reason("vp9", _webm("V_VP9", blocks)) is None


def test_webm_laced_blocks_are_skipped_until_an_unlaced_frame():
    blocks = [
        _simple_block(1, _vp9_header(0), laced=True),
        _simple_block(1, _vp9_header(1)),
    ]
    assert unsupported_reason("vp9", _webm("V_VP9", blocks)) is not None
    only_laced = [_simple_block(1, _vp9_header(0), laced=True)]
    assert unsupported_reason("vp9", _webm("V_VP9", only_laced)) is not None


def test_webm_codec_private_profile_is_checked_too():
    # Header says profile 0, CodecPrivate says 4:4:4: reject.
    private = bytes([1, 1, 0, 4, 1, 3])
    reason = unsupported_reason(
        "vp9", _webm("V_VP9", [_simple_block(1, _vp9_header(0))], private)
    )
    assert reason is not None and "chroma" in reason
    ok_private = bytes([1, 1, 0, 3, 1, 8, 4, 1, 1])
    assert (
        unsupported_reason(
            "vp9", _webm("V_VP9", [_simple_block(1, _vp9_header(0))], ok_private)
        )
        is None
    )


def test_webm_frame_header_wins_over_permissive_codec_private():
    private = bytes([1, 1, 0])
    assert (
        unsupported_reason(
            "vp9", _webm("V_VP9", [_simple_block(1, _vp9_header(1))], private)
        )
        is not None
    )


@pytest.mark.parametrize(
    "data",
    [
        _webm("V_VP9", []),  # no frame
        _webm("V_VP8", [_simple_block(1, 0x9D)]),  # wrong codec track
        _webm("V_VP9", [_simple_block(1, 0x00)]),  # bad frame marker
        b"\x1a\x45\xdf\xa3",  # magic only
        b"\x1a\x45\xdf\xa3\x00\x00\x00",  # invalid vint
    ],
)
def test_webm_unidentifiable_streams_are_rejected(data):
    assert unsupported_reason("vp9", data) is not None


def test_webm_truncated_tail_keeps_what_was_read():
    data = _webm(
        "V_VP9", [_simple_block(1, _vp9_header(0)), _simple_block(1, _vp9_header(0))]
    )
    assert unsupported_reason("vp9", data[:-5]) is None
    data = _webm(
        "V_VP9", [_simple_block(1, _vp9_header(1)), _simple_block(1, _vp9_header(0))]
    )
    assert unsupported_reason("vp9", data[:-5]) is not None


def test_webm_garbage_never_raises():
    base = _webm("V_VP9", [_simple_block(1, _vp9_header(0))])
    for i in range(len(base)):
        for flip in (0x01, 0x80, 0xFF):
            mutated = bytearray(base)
            mutated[i] ^= flip
            result = unsupported_reason("vp9", bytes(mutated))
            assert result is None or isinstance(result, str)


# --- VP9 in ISO-BMFF -------------------------------------------------------------


@pytest.mark.parametrize("profile,chroma", [(0, 0), (0, 1), (2, 1), (1, 3), (3, 3)])
def test_mp4_vp9_is_always_rejected(profile, chroma):
    """PyNvVideoCodec segfaults on VP9 from ISO-BMFF even at profile 0."""
    reason = unsupported_reason("vp9", _mp4(_vpcc(profile, chroma)))
    assert reason is not None and "Matroska" in reason
    assert unsupported_reason("vp9", _mp4(b"")) is not None


def test_mp4_stray_av1c_fourcc_in_media_data_is_ignored():
    data = _mp4(_box(b"av1C", _av1c()), b"av01") + _box(
        b"mdat", b"\xff\xff\xff\xffav1C\x81\x20\x00\x00"
    )
    assert unsupported_reason("av1", data) is None


# --- AV1 -------------------------------------------------------------------------


def test_av1_main_420_is_decodable():
    assert unsupported_reason("av1", _mp4(_box(b"av1C", _av1c()), b"av01")) is None
    assert (
        unsupported_reason("av1", _webm("V_AV1", [_simple_block(1, 0x12)], _av1c()))
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
    assert unsupported_reason("av1", _mp4(_box(b"av1C", record), b"av01")) is not None
    assert (
        unsupported_reason("av1", _webm("V_AV1", [_simple_block(1, 0x12)], record))
        is not None
    )


def test_av1_without_config_record_is_rejected():
    assert unsupported_reason("av1", _mp4(b"", b"av01")) is not None
    assert (
        unsupported_reason("av1", _webm("V_AV1", [_simple_block(1, 0x12)])) is not None
    )
