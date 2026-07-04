"""LTX-2.3 workflow helpers.

A single custom node, ``LoadImageWithEnable``, that bundles ComfyUI's stock
``LoadImage`` with an on/off ``enable`` checkbox so each keyframe lane in the
extracted Video-Gen workflow is ONE input node carrying both the image and its
toggle (instead of a LoadImage + a separate PrimitiveBoolean).

Why a custom node is the only literal way to do this:
- No stock or installed node type combines an image picker with a boolean toggle
  (verified by scanning the full /object_info of a live 0.26.0 server, 1202 types).
- ComfyUI's visual Groups can wrap the two nodes in a frame, but multiple users
  report groups silently failing to render; not reliable enough to depend on.

The upload button, preview thumbnail, and "Open in MaskEditor" UI are NOT tied to
the ``LoadImage`` node type -- they attach to any widget declared with the
``{"image_upload": True}`` metadata. ``PainterNode`` (comfy_extras/nodes_painter.py)
is a non-LoadImage node that uses exactly this metadata and gets the full UI, which
is the proof this custom node inherits it too. So there is no mask-editor / preview
regression here.

To install: copy (or symlink) this whole ``ltx23_helpers`` directory into your
ComfyUI ``custom_nodes/`` folder and restart ComfyUI.
"""

import os

import folder_paths


class LoadImageWithEnable:
    """LoadImage + an ``enable`` checkbox, as a single node.

    Outputs ``IMAGE`` and ``MASK`` exactly like the core ``LoadImage`` (image is
    ALWAYS loaded and returned, so downstream consumers that are permanently
    wired don't break when the lane is disabled), plus ``BOOLEAN`` reflecting the
    ``enable`` state -- the boolean fans into the lazy Switch nodes that gate
    whether the keyframe actually conditions the video.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Mirror core LoadImage's input declaration EXACTLY (same image_upload
        # metadata) so the frontend attaches upload + preview + mask-editor to it.
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
                "enable": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOOLEAN")
    RETURN_NAMES = ("IMAGE", "MASK", "ENABLE")
    FUNCTION = "load"
    CATEGORY = "LTX23"
    SEARCH_ALIASES = ["load image with toggle", "keyframe", "image with enable"]

    # Delegate the actual decode to the core LoadImage so this stays robust
    # against upstream changes to the image-loading internals.
    def load(self, image, enable):
        from nodes import LoadImage

        img, mask = LoadImage().load_image(image)
        return (img, mask, bool(enable))


NODE_CLASS_MAPPINGS = {"LoadImageWithEnable": LoadImageWithEnable}
NODE_DISPLAY_NAME_MAPPINGS = {"LoadImageWithEnable": "Load Image + Enable"}
