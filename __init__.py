"""MiniMax H3 Motion Context / MultiRef / latent-masking custom nodes.

ComfyUI provides ``folder_paths`` before loading custom nodes.  Keeping the
package import harmless when that module is absent lets the repository's
standalone CPU/static tests run without pretending a full ComfyUI installation
is available.
"""

try:
    import folder_paths as _folder_paths  # noqa: F401
except ModuleNotFoundError:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
else:
    # Pytest may collect this file directly as ``__init__`` rather than as a
    # package.  In that mode relative imports are impossible; real ComfyUI
    # always supplies a package name when loading a custom-node directory.
    if not __package__:
        NODE_CLASS_MAPPINGS = {}
        NODE_DISPLAY_NAME_MAPPINGS = {}
    else:
        from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./js"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
