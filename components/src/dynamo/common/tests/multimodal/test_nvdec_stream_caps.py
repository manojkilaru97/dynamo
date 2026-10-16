# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Omni pre-NVDEC safety check (PyNvVideoCodec demuxer faked)."""

import ctypes
import sys
import types

import pytest

import dynamo.common.multimodal.nvdec_stream_caps as caps
from dynamo.common.multimodal.nvdec_stream_caps import (
    check_vp9_packet,
    unsupported_reason,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]

# First byte of a VP9 frame as written by libvpx, per profile (literal values,
# not derived from the parser's bit layout, so a swapped profile bit is caught).
KEY = {0: 0x82, 1: 0xA2, 2: 0x92, 3: 0xB1}
INTER = {0: 0x86, 1: 0xA6, 2: 0x96, 3: 0xB5}


def frame(first: int, size: int = 12) -> bytes:
    return bytes([first]) + b"\x10" * (size - 1)


def superframe(*frames: bytes, width: int = 1) -> bytes:
    marker = 0xC0 | ((width - 1) << 3) | (len(frames) - 1)
    index = b"".join(len(f).to_bytes(width, "little") for f in frames)
    return b"".join(frames) + bytes([marker]) + index + bytes([marker])


class FakePacket:
    def __init__(self, data: bytes):
        self._buf = ctypes.create_string_buffer(data, len(data))
        self.bsl = len(data)
        self.bsl_data = ctypes.addressof(self._buf) if data else 0


class FakeDemuxer:
    def __init__(self, codec, container, packets, fail_after=None):
        self._codec, self._container = codec, container
        self._packets = list(packets)
        self._fail_after = fail_after
        self.demuxed = 0

    def GetNvCodecId(self):
        return self._codec

    def GetContainerName(self):
        return self._container

    def Demux(self):
        if self._fail_after is not None and self.demuxed >= self._fail_after:
            raise RuntimeError("demux error")
        self.demuxed += 1
        return FakePacket(self._packets.pop(0) if self._packets else b"")


def fake_nvc(
    monkeypatch,
    codec="VP9",
    container="matroska,webm",
    packets=(),
    open_error=False,
    fail_after=None,
):
    mod = types.ModuleType("PyNvVideoCodec")
    mod.cudaVideoCodec = types.SimpleNamespace(
        VP9="VP9", AV1="AV1", H264="H264", HEVC="HEVC", VP8="VP8"
    )
    state = {}

    def create(path):
        if open_error:
            raise RuntimeError("cannot open")
        state["demuxer"] = FakeDemuxer(codec, container, packets, fail_after)
        return state["demuxer"]

    mod.CreateDemuxer = create
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", mod)
    return state


# --- frame/packet parsing ------------------------------------------------------------


@pytest.mark.parametrize("profile", [0, 2])
def test_420_profiles_pass(profile):
    assert check_vp9_packet(frame(KEY[profile])) == 1
    assert check_vp9_packet(frame(INTER[profile])) == 1


@pytest.mark.parametrize("profile", [1, 3])
def test_non_420_profiles_fail(profile):
    for first in (KEY[profile], INTER[profile]):
        with pytest.raises(caps._Unsupported, match=f"profile {profile}"):
            check_vp9_packet(frame(first))


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_every_superframe_frame_is_checked(width):
    ok = superframe(frame(KEY[0]), frame(INTER[0]), frame(INTER[0]), width=width)
    assert check_vp9_packet(ok) == 3
    hidden = superframe(frame(KEY[0]), frame(INTER[0]), frame(INTER[1]), width=width)
    with pytest.raises(caps._Unsupported, match="profile 1"):
        check_vp9_packet(hidden)


def test_superframe_zero_size_entries_are_skipped():
    data = superframe(frame(KEY[0]), frame(INTER[0]))
    # Rewrite the index as (n0, 0, n1): every non-empty frame is still checked.
    marker = 0xC2
    body = frame(KEY[0]) + frame(INTER[1])
    bad = body + bytes([marker, 12, 0, 12, marker])
    with pytest.raises(caps._Unsupported, match="profile 1"):
        check_vp9_packet(bad)
    assert check_vp9_packet(data) == 2


@pytest.mark.parametrize(
    "index",
    [
        bytes([0xC1, 0xFF, 0xFF, 0xC1]),  # sizes overflow the payload
        bytes([0xC1, 0x00, 0x00, 0xC1]),  # all entries empty
    ],
)
def test_inconsistent_superframe_index_fails_closed(index):
    """libvpx decodes the entries that fit; never fall back to one frame."""
    with pytest.raises(caps._Unsupported, match="superframe index"):
        check_vp9_packet(frame(KEY[0], 6) + index)


def test_overflowing_index_cannot_hide_a_profile1_frame():
    body = frame(KEY[0]) + frame(KEY[1])
    with pytest.raises(caps._Unsupported):
        check_vp9_packet(body + bytes([0xCA, 12, 0, 12, 0, 0xFF, 0xFF, 0xCA]))


@pytest.mark.parametrize("data", [b"", frame(0x00), frame(0x40)])
def test_invalid_packets_fail(data):
    with pytest.raises(caps._Unsupported):
        check_vp9_packet(data)


# --- decisions from the demuxer --------------------------------------------------------


def test_vp9_matroska_all_420_is_decodable(monkeypatch):
    packets = [
        frame(KEY[0]),
        frame(INTER[0]),
        superframe(frame(INTER[0]), frame(INTER[0])),
    ]
    state = fake_nvc(monkeypatch, packets=packets)
    assert unsupported_reason(b"x", strict=True) is None
    # Read to the end: the end is three empty reads in a row.
    assert state["demuxer"].demuxed == len(packets) + caps._EOF_EMPTY_READS


@pytest.mark.parametrize("strict", [True, False])
def test_empty_packet_mid_stream_is_not_end_of_stream(monkeypatch, strict):
    """An empty block must not end the scan before a later profile-1 frame."""
    fake_nvc(monkeypatch, packets=[frame(KEY[0]), b"", b"", frame(KEY[1])])
    assert "profile 1" in unsupported_reason(b"x", strict=strict)


@pytest.mark.parametrize("strict", [True, False])
def test_unsafe_superframe_in_a_later_packet_is_rejected(monkeypatch, strict):
    hidden = superframe(frame(INTER[0]), frame(INTER[1]), frame(INTER[0]))
    fake_nvc(monkeypatch, packets=[frame(KEY[0]), frame(INTER[0]), hidden])
    assert "profile 1" in unsupported_reason(b"x", strict=strict)


@pytest.mark.parametrize("strict", [True, False])
def test_vp9_profile1_anywhere_is_rejected(monkeypatch, strict):
    """VP9 carries the profile per frame; a late profile-1 keyframe counts."""
    fake_nvc(monkeypatch, packets=[frame(KEY[0])] * 5 + [frame(KEY[1])])
    assert "profile 1" in unsupported_reason(b"x", strict=strict)


@pytest.mark.parametrize("container", ["mov,mp4,m4a,3gp,3g2,mj2", "mp4", "avi", ""])
@pytest.mark.parametrize("strict", [True, False])
def test_vp9_outside_matroska_is_rejected(monkeypatch, container, strict):
    """PyNvVideoCodec segfaults on VP9 read from MP4 even at profile 0."""
    state = fake_nvc(monkeypatch, container=container, packets=[frame(KEY[0])])
    assert "Matroska" in unsupported_reason(b"x", strict=strict)
    assert state["demuxer"].demuxed == 0


@pytest.mark.parametrize("strict", [True, False])
def test_av1_is_rejected(monkeypatch, strict):
    fake_nvc(monkeypatch, codec="AV1", container="matroska,webm")
    assert "AV1" in unsupported_reason(b"x", strict=strict)


@pytest.mark.parametrize("codec", ["H264", "HEVC"])
@pytest.mark.parametrize(
    "container", ["avi", "mov,mp4,m4a,3gp,3g2,mj2", "matroska,webm"]
)
@pytest.mark.parametrize("strict", [True, False])
def test_h264_hevc_keep_nvdec(monkeypatch, codec, container, strict):
    state = fake_nvc(
        monkeypatch, codec=codec, container=container, packets=[b"\x00" * 8]
    )
    assert unsupported_reason(b"x", strict=strict) is None
    assert state["demuxer"].demuxed == 0  # packets are only read for VP9


@pytest.mark.parametrize("strict", [True, False])
def test_vp8_only_from_matroska(monkeypatch, strict):
    fake_nvc(monkeypatch, codec="VP8", container="matroska,webm")
    assert unsupported_reason(b"x", strict=strict) is None
    fake_nvc(monkeypatch, codec="VP8", container="mov,mp4,m4a,3gp,3g2,mj2")
    assert "Matroska" in unsupported_reason(b"x", strict=strict)


def test_unopenable_stream_fails_closed_only_when_strict(monkeypatch):
    fake_nvc(monkeypatch, open_error=True)
    assert "could not open" in unsupported_reason(b"x", strict=True)
    assert unsupported_reason(b"x", strict=False) is None


def test_vp9_without_frames_is_rejected(monkeypatch):
    fake_nvc(monkeypatch, packets=[])
    assert unsupported_reason(b"x", strict=False) == "no VP9 frame found"


def test_vp9_demux_error_is_rejected(monkeypatch):
    fake_nvc(monkeypatch, packets=[frame(KEY[0])] * 3, fail_after=2)
    assert "could not be inspected" in unsupported_reason(b"x", strict=False)


def test_vp9_packet_budget_is_enforced(monkeypatch):
    monkeypatch.setattr(caps, "_MAX_PACKETS", 4)
    state = fake_nvc(monkeypatch, packets=[frame(KEY[0])] * 10, fail_after=4)
    assert "too long" in unsupported_reason(b"x", strict=False)
    assert state["demuxer"].demuxed == 4  # a fifth Demux() call would raise
