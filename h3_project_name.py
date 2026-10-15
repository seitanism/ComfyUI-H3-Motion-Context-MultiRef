"""Single project-name source for all music-video files and directories."""

from __future__ import annotations

import re


def normalize_project_name(value: str) -> str:
    """Return a filesystem-safe, single-directory project identifier."""
    name = str(value or "").strip()
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_+", "_", name).strip(" ._")
    if not name or name in {".", ".."}:
        raise ValueError("h3_project_name: enter a non-empty project name")
    return name


class MiniMaxH3MusicVideoProjectName:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project_name": (
                    "STRING",
                    {
                        "default": "my_music_video",
                        "multiline": False,
                        "tooltip": (
                            "Enter this once. It names the checkpoint set, preview folder "
                            "and final MP4 prefix for the whole music-video workflow."
                        ),
                    },
                )
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("project_name",)
    FUNCTION = "resolve"
    CATEGORY = "conditioning/minimax/music video"
    DESCRIPTION = "One project name propagated to every music-video directory and MP4 output."

    @classmethod
    def IS_CHANGED(cls, project_name):
        return normalize_project_name(project_name)

    def resolve(self, project_name):
        return (normalize_project_name(project_name),)
