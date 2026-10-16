# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

import dynamo.common.multimodal.video_loader as video_loader_module
from dynamo.common.http import HttpStatusError
from dynamo.common.http.url_validator import UrlValidationError, UrlValidationPolicy
from dynamo.common.multimodal.codec_errors import MissingMediaDecoderError
from dynamo.common.multimodal.video_loader import VideoLoader
from dynamo.common.utils.install_media_decoders import VALIDATED_SPECS

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


@pytest.mark.asyncio
async def test_load_video_rejects_http_by_default():
    """Wiring smoke: VideoLoader plumbs ``url_policy`` to the validator.

    Validator behavior is covered in ``test_url_validator.py``;
    per-hop SSRF revalidation in ``http/test_http_backends.py``.
    """
    loader = VideoLoader(url_policy=UrlValidationPolicy())

    with pytest.raises(UrlValidationError, match="not allowed"):
        await loader.load_video("http://example.com/x.mp4")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client_error",
    [
        UrlValidationError("blocked host"),
        HttpStatusError(415, "Unsupported Media Type", "https://example.com/x.mp4"),
    ],
)
async def test_load_video_preserves_client_error(client_error):
    loader = VideoLoader()
    loader._load_video_with_vllm = AsyncMock(  # type: ignore[method-assign]
        side_effect=client_error
    )

    with pytest.raises(type(client_error)) as exc_info:
        await loader.load_video("https://example.com/x.mp4")

    assert exc_info.value is client_error


@pytest.mark.asyncio
async def test_load_video_uses_vllm_media_connector():
    loader = VideoLoader()
    # data: scheme is in the default allowlist regardless of env flags.
    loader._url_policy = UrlValidationPolicy()
    frames = np.arange(24, dtype=np.uint8).reshape(1, 2, 4, 3)[:, :, ::-1, :]
    metadata = {"fps": 4.0, "frames_indices": [0], "total_num_frames": 1}
    loader._load_video_with_vllm = AsyncMock(  # type: ignore[method-assign]
        return_value=(frames, metadata)
    )

    loaded_frames, loaded_metadata = await loader.load_video(
        "data:video/webm;base64,Zm9v"
    )

    assert loaded_frames.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(loaded_frames, np.ascontiguousarray(frames))
    assert loaded_metadata == metadata


@pytest.mark.asyncio
async def test_load_video_batch_uses_url_loader():
    loader = VideoLoader()
    first = (
        np.zeros((1, 2, 2, 3), dtype=np.uint8),
        {"fps": 2.0, "frames_indices": [0], "total_num_frames": 1},
    )
    second = (
        np.ones((1, 2, 2, 3), dtype=np.uint8),
        {"fps": 2.0, "frames_indices": [0], "total_num_frames": 1},
    )
    loader.load_video = AsyncMock(side_effect=[first, second])  # type: ignore[method-assign]

    videos = await loader.load_video_batch(
        [
            {"Url": "https://example.com/one.mp4"},
            {"Url": "https://example.com/two.mp4"},
        ]
    )

    np.testing.assert_array_equal(videos[0][0], first[0])
    np.testing.assert_array_equal(videos[1][0], second[0])
    assert videos[0][1] == first[1]
    assert videos[1][1] == second[1]


@pytest.mark.asyncio
async def test_load_video_batch_rejects_decoded_variant_without_frontend_decoding():
    loader = VideoLoader(enable_frontend_decoding=False)

    with pytest.raises(ValueError, match="enable_frontend_decoding=False"):
        await loader.load_video_batch([{"Decoded": {"shape": [1, 2, 2, 3]}}])


@pytest.mark.asyncio
async def test_load_video_batch_prioritizes_typed_client_error():
    loader = VideoLoader()
    client_error = HttpStatusError(
        415, "Unsupported Media Type", "https://example.com/bad.mp4"
    )
    loader.load_video = AsyncMock(  # type: ignore[method-assign]
        side_effect=[RuntimeError("decode failed"), client_error]
    )

    with pytest.raises(HttpStatusError) as exc_info:
        await loader.load_video_batch(
            [
                {"Url": "https://example.com/bad.mp4"},
                {"Url": "https://example.com/unsupported.mp4"},
            ]
        )

    assert exc_info.value is client_error


@pytest.mark.asyncio
async def test_load_video_batch_reads_decoded_variant_with_metadata(monkeypatch):
    loader = VideoLoader(enable_frontend_decoding=False)
    loader._enable_frontend_decoding = True
    loader._nixl_connector = object()

    decoded_item = {
        "shape": [1, 2, 2, 3],
        "metadata": {"fps": 3.0, "frames_indices": [0], "total_num_frames": 1},
    }
    frames = np.arange(12, dtype=np.uint8).reshape(1, 2, 2, 3)
    read_decoded = AsyncMock(return_value=(frames, decoded_item["metadata"]))
    monkeypatch.setattr(
        video_loader_module, "read_decoded_media_via_nixl", read_decoded
    )

    videos = await loader.load_video_batch([{"Decoded": decoded_item}])

    np.testing.assert_array_equal(videos[0][0], np.ascontiguousarray(frames))
    assert videos[0][1] == decoded_item["metadata"]
    read_decoded.assert_awaited_once_with(
        loader._nixl_connector,
        decoded_item,
        return_metadata=True,
    )


@pytest.mark.asyncio
async def test_decode_video_bytes_routes_h264_to_nvdec(monkeypatch):
    loader = VideoLoader()
    frames = np.zeros((2, 4, 6, 3), dtype=np.uint8)
    meta = {"fps": 30.0, "frames_indices": [0, 1], "total_num_frames": 2}
    monkeypatch.setattr(video_loader_module, "probe_video_codec", lambda b: "h264")
    monkeypatch.setattr(video_loader_module, "should_use_nvdec", lambda c: True)
    monkeypatch.setattr(
        video_loader_module, "decode_video_nvdec", lambda b, n: (frames, meta)
    )
    media_io = _RecordingMediaIO(frames)

    got_frames, got_meta = await loader._decode_video_bytes(b"h264-bytes", media_io)

    np.testing.assert_array_equal(got_frames, frames)
    assert got_meta == meta
    assert media_io.calls == []  # hardware decoded; software untouched


@pytest.mark.asyncio
async def test_decode_video_bytes_routes_royalty_free_to_software(monkeypatch):
    loader = VideoLoader()
    frames = np.zeros((1, 2, 2, 3), dtype=np.uint8)
    monkeypatch.setattr(video_loader_module, "probe_video_codec", lambda b: "vp9")
    monkeypatch.setattr(video_loader_module, "should_use_nvdec", lambda c: False)
    called = {"decode": False}

    def _decode(*a, **k):
        called["decode"] = True

    monkeypatch.setattr(video_loader_module, "decode_video_nvdec", _decode)
    media_io = _RecordingMediaIO(frames)

    result = await loader._decode_video_bytes(b"vp9-bytes", media_io)

    assert result is media_io.result  # VP9 stays on the software path
    assert called["decode"] is False  # NVDEC not invoked
    assert media_io.calls == [b"vp9-bytes"]


@pytest.mark.asyncio
async def test_decode_video_bytes_falls_back_on_nvdec_failure(monkeypatch):
    loader = VideoLoader()
    frames = np.zeros((1, 2, 2, 3), dtype=np.uint8)
    monkeypatch.setattr(video_loader_module, "probe_video_codec", lambda b: "hevc")
    monkeypatch.setattr(video_loader_module, "should_use_nvdec", lambda c: True)

    def _boom(*a, **k):
        raise RuntimeError("nvdec session limit")

    monkeypatch.setattr(video_loader_module, "decode_video_nvdec", _boom)
    media_io = _RecordingMediaIO(frames)

    result = await loader._decode_video_bytes(b"hevc-bytes", media_io)

    assert result is media_io.result  # NVDEC failure falls back, never raises
    assert media_io.calls == [b"hevc-bytes"]


class _RecordingMediaIO:
    """Stub for vLLM's VideoMediaIO: records load_bytes calls."""

    def __init__(self, frames):
        self.result = (frames, {"fps": 1.0})
        self.calls: list[bytes] = []

    def load_bytes(self, content: bytes):
        self.calls.append(content)
        return self.result


class _ImportErrorMediaIO:
    """Stub reproducing vLLM's lazy cv2 import failure on a stripped image."""

    def load_bytes(self, content: bytes):
        raise ModuleNotFoundError("No module named 'cv2'", name="cv2")


@pytest.mark.asyncio
async def test_decode_video_bytes_missing_decoder_is_actionable(monkeypatch):
    """A bare `No module named 'cv2'` must become the actionable codec error.

    Reproduced on a real runtime image before this existed: the user-visible
    text was `Failed to load video from ...: No module named 'cv2'` -- no
    codec, no remedy.
    """
    loader = VideoLoader()
    monkeypatch.setattr(video_loader_module, "probe_video_codec", lambda b: "vp9")
    monkeypatch.setattr(video_loader_module, "should_use_nvdec", lambda c: False)

    with pytest.raises(MissingMediaDecoderError) as exc_info:
        await loader._decode_video_bytes(b"vp9-bytes", _ImportErrorMediaIO())

    msg = str(exc_info.value)
    assert "'vp9'" in msg  # names the codec
    assert VALIDATED_SPECS["opencv-python-headless"] in msg  # bounded spec
    assert "install_media_decoders vllm" in msg  # installer command
    assert "cv2" in msg


@pytest.mark.asyncio
async def test_load_video_batch_preserves_missing_decoder_error():
    """The batch aggregate wraps failures in a generic Exception; the
    missing-decoder type must survive it (review finding)."""
    loader = VideoLoader()
    err = video_loader_module.video_decoder_missing(
        "vllm", "opencv-python-headless", "cv2", "vp9"
    )
    loader.load_video = AsyncMock(side_effect=err)  # type: ignore[method-assign]

    with pytest.raises(MissingMediaDecoderError) as exc_info:
        await loader.load_video_batch([{"Url": "https://example.com/x.mp4"}])

    assert exc_info.value is err


@pytest.mark.asyncio
async def test_load_video_preserves_missing_decoder_error(monkeypatch):
    """The generic ValueError wrap must not erase the decoder-missing type.

    A missing decoder is deployment configuration, not a bad request, so it
    must not degrade into the ValueError that handlers map to a client error.
    """
    loader = VideoLoader()
    err = video_loader_module.video_decoder_missing(
        "vllm", "opencv-python-headless", "cv2", "vp9"
    )
    loader._load_video_with_vllm = AsyncMock(  # type: ignore[method-assign]
        side_effect=err
    )

    with pytest.raises(MissingMediaDecoderError) as exc_info:
        await loader.load_video("https://example.com/x.mp4")

    assert exc_info.value is err


@pytest.mark.asyncio
async def test_load_video_without_keep_bytes_leaves_metadata_untouched(monkeypatch):
    """The default (Super 3.5 / non-audio models) contract: no source bytes."""
    loader = VideoLoader(url_policy=UrlValidationPolicy())
    frames = np.zeros((1, 2, 2, 3), dtype=np.uint8)
    metadata = {"fps": 4.0, "frames_indices": [0], "total_num_frames": 1}
    loader._decode_video_bytes = AsyncMock(  # type: ignore[method-assign]
        return_value=(frames, metadata)
    )

    _, loaded_metadata = await loader.load_video("data:video/mp4;base64,Zm9vYmFy")

    assert loaded_metadata == metadata
    assert "original_video_bytes" not in loaded_metadata
    loader._decode_video_bytes.assert_awaited_once()
    assert loader._decode_video_bytes.await_args.args[0] == b"foobar"


@pytest.mark.asyncio
async def test_load_video_keep_bytes_attaches_original_video_bytes():
    loader = VideoLoader(url_policy=UrlValidationPolicy())
    frames = np.zeros((1, 2, 2, 3), dtype=np.uint8)
    metadata = {"fps": 4.0, "frames_indices": [0], "total_num_frames": 1}
    loader._decode_video_bytes = AsyncMock(  # type: ignore[method-assign]
        return_value=(frames, metadata)
    )

    _, loaded_metadata = await loader.load_video(
        "data:video/webm;base64,Zm9vYmFy", keep_video_bytes=True
    )

    assert loaded_metadata["original_video_bytes"] == b"foobar"
    assert {
        k: v for k, v in loaded_metadata.items() if k != "original_video_bytes"
    } == (metadata)
    assert "original_video_bytes" not in metadata  # decoder output not mutated


@pytest.mark.asyncio
async def test_load_video_batch_forwards_keep_bytes_only_when_requested():
    loader = VideoLoader()
    video = (np.zeros((1, 2, 2, 3), dtype=np.uint8), {"fps": 2.0})
    loader.load_video = AsyncMock(return_value=video)  # type: ignore[method-assign]
    items = [{"Url": "https://example.com/one.mp4"}]

    await loader.load_video_batch(items)
    loader.load_video.assert_awaited_once_with("https://example.com/one.mp4")

    loader.load_video.reset_mock()
    await loader.load_video_batch(items, keep_video_bytes=True)
    loader.load_video.assert_awaited_once_with(
        "https://example.com/one.mp4", keep_video_bytes=True
    )


@pytest.mark.asyncio
async def test_load_video_batch_keep_bytes_rejects_frontend_decoded_video():
    loader = VideoLoader()

    with pytest.raises(ValueError, match="frontend-decoded"):
        await loader.load_video_batch(
            [{"Decoded": {"shape": [1, 2, 2, 3]}}], keep_video_bytes=True
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("codec", ["vp9", "av1", "vp8"])
async def test_nvdec_royalty_free_codecs_opt_in_routes_to_nvdec(monkeypatch, codec):
    loader = VideoLoader(nvdec_royalty_free_codecs=True)
    frames = np.zeros((1, 2, 2, 3), dtype=np.uint8)
    meta = {"fps": 4.0, "frames_indices": [0], "total_num_frames": 1}
    monkeypatch.setattr(video_loader_module, "probe_video_codec", lambda b: codec)
    monkeypatch.setattr(video_loader_module, "should_use_nvdec", lambda c: False)
    monkeypatch.setattr(video_loader_module, "nvdec_available", lambda: True)
    monkeypatch.setattr(video_loader_module, "_nvdec_safe_for_omni", lambda c, b: True)
    monkeypatch.setattr(
        video_loader_module, "decode_video_nvdec", lambda b, n: (frames, meta)
    )
    media_io = _RecordingMediaIO(frames)

    _, got_meta = await loader._decode_video_bytes(b"bytes", media_io)

    assert got_meta == meta
    assert media_io.calls == []


# First byte of a libvpx VP9 keyframe per profile.
_VP9_KEYFRAME = {0: 0x82, 1: 0xA2, 2: 0x92, 3: 0xB1}


def _fake_demuxer(monkeypatch, codec, container="matroska,webm", packets=()):
    """Install a fake PyNvVideoCodec whose demuxer reports ``codec``/``packets``."""
    import ctypes
    import sys
    import types

    mod = types.ModuleType("PyNvVideoCodec")
    mod.cudaVideoCodec = types.SimpleNamespace(
        VP9="VP9", AV1="AV1", H264="H264", HEVC="HEVC", VP8="VP8"
    )
    queue = list(packets)

    class Packet:
        def __init__(self, data):
            self._buf = ctypes.create_string_buffer(data, len(data))
            self.bsl, self.bsl_data = len(data), ctypes.addressof(self._buf)

    class Demuxer:
        def GetNvCodecId(self):
            return codec

        def GetContainerName(self):
            return container

        def Demux(self):
            return Packet(queue.pop(0) if queue else b"")

    mod.CreateDemuxer = lambda path: Demuxer()
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", mod)


def _nvdec_mock(monkeypatch):
    frames = np.zeros((1, 2, 2, 3), dtype=np.uint8)
    meta = {"fps": 4.0, "frames_indices": [0], "total_num_frames": 1}
    decode = Mock(return_value=(frames, meta))
    monkeypatch.setattr(video_loader_module, "decode_video_nvdec", decode)
    monkeypatch.setattr(video_loader_module, "nvdec_available", lambda: True)
    return decode, frames, meta


@pytest.mark.asyncio
@pytest.mark.parametrize("hint", ["vp9", "h264"])
@pytest.mark.parametrize("profile", [1, 3])
async def test_omni_vp9_non_420_profile_never_reaches_nvdec(monkeypatch, hint, profile):
    """PyNvVideoCodec segfaults the worker on VP9 profiles 1/3 (4:2:2/4:4:4),
    also when stray bytes make the byte probe say H.264."""
    decode, frames, _ = _nvdec_mock(monkeypatch)
    monkeypatch.setattr(video_loader_module, "probe_video_codec", lambda b: hint)
    monkeypatch.setattr(video_loader_module, "should_use_nvdec", lambda c: c == "h264")
    _fake_demuxer(monkeypatch, "VP9", packets=[bytes([_VP9_KEYFRAME[profile]]) * 8])
    loader = VideoLoader(nvdec_royalty_free_codecs=True)
    media_io = _RecordingMediaIO(frames)

    result = await loader._decode_video_bytes(b"clip", media_io)

    decode.assert_not_called()
    assert result is media_io.result  # software path (an error on stripped images)
    assert media_io.calls == [b"clip"]


@pytest.mark.asyncio
async def test_omni_vp9_in_mp4_never_reaches_nvdec(monkeypatch):
    decode, frames, _ = _nvdec_mock(monkeypatch)
    monkeypatch.setattr(video_loader_module, "probe_video_codec", lambda b: "vp9")
    _fake_demuxer(monkeypatch, "VP9", container="mov,mp4,m4a,3gp,3g2,mj2")

    await VideoLoader(nvdec_royalty_free_codecs=True)._decode_video_bytes(
        b"clip", _RecordingMediaIO(frames)
    )

    decode.assert_not_called()


@pytest.mark.asyncio
async def test_omni_check_error_fails_closed(monkeypatch):
    """An error inside the check (e.g. a full /tmp) must not reach NVDEC."""
    decode, frames, _ = _nvdec_mock(monkeypatch)
    monkeypatch.setattr(video_loader_module, "probe_video_codec", lambda b: "h264")
    monkeypatch.setattr(video_loader_module, "should_use_nvdec", lambda c: True)
    import dynamo.common.multimodal.nvdec_stream_caps as caps

    def broken(data, strict):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(caps, "unsupported_reason", broken)
    media_io = _RecordingMediaIO(frames)

    result = await VideoLoader(nvdec_royalty_free_codecs=True)._decode_video_bytes(
        b"clip", media_io
    )

    decode.assert_not_called()
    assert result is media_io.result


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", [0, 2])
async def test_omni_vp9_420_profile_still_uses_nvdec(monkeypatch, profile):
    decode, frames, meta = _nvdec_mock(monkeypatch)
    monkeypatch.setattr(video_loader_module, "probe_video_codec", lambda b: "vp9")
    _fake_demuxer(monkeypatch, "VP9", packets=[bytes([_VP9_KEYFRAME[profile]]) * 8])
    media_io = _RecordingMediaIO(frames)

    _, got_meta = await VideoLoader(nvdec_royalty_free_codecs=True)._decode_video_bytes(
        b"clip", media_io
    )

    assert got_meta == meta
    decode.assert_called_once()
    assert media_io.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("hint", ["av1", "h264", "vp9"])
async def test_omni_real_h264_keeps_nvdec_whatever_the_hint(monkeypatch, hint):
    """A stray ``av01``/``vp09`` marker in an H.264 file must not cost NVDEC."""
    decode, frames, meta = _nvdec_mock(monkeypatch)
    monkeypatch.setattr(video_loader_module, "probe_video_codec", lambda b: hint)
    monkeypatch.setattr(video_loader_module, "should_use_nvdec", lambda c: c == "h264")
    _fake_demuxer(monkeypatch, "H264", container="avi")

    _, got_meta = await VideoLoader(nvdec_royalty_free_codecs=True)._decode_video_bytes(
        b"clip", _RecordingMediaIO(frames)
    )

    assert got_meta == meta
    decode.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("codec", ["h264", "hevc", "vp9", "vp8", "av1", None])
async def test_default_loader_never_runs_the_omni_check(monkeypatch, codec):
    """Super 3.5 / Super 3 (default loader): routing exactly as before."""
    decode, frames, meta = _nvdec_mock(monkeypatch)
    monkeypatch.setattr(video_loader_module, "probe_video_codec", lambda b: codec)
    monkeypatch.setattr(
        video_loader_module, "should_use_nvdec", lambda c: c in ("h264", "hevc")
    )
    check = Mock(return_value=False)
    monkeypatch.setattr(video_loader_module, "_nvdec_safe_for_omni", check)
    media_io = _RecordingMediaIO(frames)

    result = await VideoLoader()._decode_video_bytes(b"bytes", media_io)

    check.assert_not_called()
    if codec in ("h264", "hevc"):
        decode.assert_called_once()
        assert result == (frames, meta)
    else:
        decode.assert_not_called()
        assert media_io.calls == [b"bytes"]


def test_default_loader_never_imports_the_omni_check():
    """Run in a fresh interpreter so nothing imported earlier hides an import."""
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        """
        import asyncio, sys
        import dynamo.common.multimodal.video_loader as vl

        decoded = []
        vl.nvdec_available = lambda: True
        vl.should_use_nvdec = lambda codec: codec in ("h264", "hevc")
        vl.decode_video_nvdec = lambda data, n: decoded.append(data) or ("nvdec", {})

        class MediaIO:
            def load_bytes(self, content):
                return ("software", content)

        clips = (b"avc1" + bytes(64), b"hvc1" + bytes(64), b"V_VP9" + bytes(64))
        for data in clips:
            asyncio.run(vl.VideoLoader()._decode_video_bytes(data, MediaIO()))
        assert decoded == list(clips[:2]), decoded  # H.264/HEVC took the NVDEC branch
        assert "dynamo.common.multimodal.nvdec_stream_caps" not in sys.modules
        print("ok")
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().endswith("ok")


@pytest.mark.asyncio
async def test_nvdec_royalty_free_codecs_default_keeps_software_path(monkeypatch):
    """Default (Super 3.5): VP9 never goes to NVDEC, as before."""
    loader = VideoLoader()
    frames = np.zeros((1, 2, 2, 3), dtype=np.uint8)
    monkeypatch.setattr(video_loader_module, "probe_video_codec", lambda b: "vp9")
    monkeypatch.setattr(video_loader_module, "should_use_nvdec", lambda c: False)
    monkeypatch.setattr(video_loader_module, "nvdec_available", lambda: True)

    def _never(*a, **k):
        raise AssertionError("NVDEC must not be used for VP9 by default")

    monkeypatch.setattr(video_loader_module, "decode_video_nvdec", _never)
    media_io = _RecordingMediaIO(frames)

    result = await loader._decode_video_bytes(b"vp9-bytes", media_io)

    assert result is media_io.result
    assert media_io.calls == [b"vp9-bytes"]
