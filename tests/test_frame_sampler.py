from __future__ import annotations

from pathlib import Path

from vlm_pipeline.frame_sampler import build_ffmpeg_command


def test_build_ffmpeg_command_contains_expected_args():
    cmd = build_ffmpeg_command(
        input_video=Path("input.mp4"),
        output_pattern=Path("frames/frame_%04d.jpg"),
        start_s=1.25,
        end_s=4.75,
        fps=1.0,
        max_frames=16,
    )

    assert cmd[:4] == ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    assert "-ss" in cmd
    assert "1.250000" in cmd
    assert "-vf" in cmd
    assert "fps=1.000000" in cmd
    assert "-frames:v" in cmd
    assert "16" in cmd
    assert cmd[-1] == "frames/frame_%04d.jpg"
