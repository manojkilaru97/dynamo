# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU integration test for the Omni pre-NVDEC check against the real PyNvVideoCodec.

The unit tests fake PyNvVideoCodec. This one runs its real demuxer on two committed
VP9 WebM clips, so an API or enum change in a PyNvVideoCodec upgrade cannot silently
reopen the crash: profile 1 (4:4:4) must be refused, profile 0 allowed. It never
creates a decoder, so even a regression cannot crash the test process. Skips where
PyNvVideoCodec/NVDEC is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dynamo.common.multimodal import nvdec_decoder as nd

pytestmark = [
    pytest.mark.integration,
    pytest.mark.post_merge,
    pytest.mark.gpu_1,
    pytest.mark.vllm,
]

_MEDIA_DIR = (
    Path(__file__).resolve().parents[6] / "lib" / "llm" / "tests" / "data" / "media"
)


def _clip(name: str) -> bytes:
    if not nd.nvdec_available():
        pytest.skip("PyNvVideoCodec/NVDEC not available (needs the video capability)")
    path = _MEDIA_DIR / name
    if not path.is_file():
        pytest.skip(f"fixture not available: {path}")
    return path.read_bytes()


@pytest.mark.parametrize("strict", [True, False])
def test_real_demuxer_refuses_vp9_profile1(strict):
    from dynamo.common.multimodal.nvdec_stream_caps import unsupported_reason

    reason = unsupported_reason(_clip("64p_5_vp9_p1_444.webm"), strict=strict)
    assert reason is not None and "profile 1" in reason


@pytest.mark.parametrize("strict", [True, False])
def test_real_demuxer_allows_vp9_profile0(strict):
    from dynamo.common.multimodal.nvdec_stream_caps import unsupported_reason

    assert unsupported_reason(_clip("64p_5_vp9_p0.webm"), strict=strict) is None
