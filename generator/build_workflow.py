#!/usr/bin/env python3
"""
Generator: 3-meta-stage ComfyUI workflow.

  [ Global Resolution ] -> [ Image Generation ] -> [ Inpainting ] -> [ Video Generation ]

  * Image Generation: 3 lanes (first/mid/last), each OPTIONAL (gen toggle), each with
    positive + negative prompt and an OPTIONAL reference image (style/composition via img2img),
    all sharing one global resolution.
  * Inpainting: 3 lanes, each OPTIONAL (inpaint toggle, default off). Pass-through when off.
  * Video Generation: LTX-2.3 two-stage distilled (native audio). Uses a frame only if its gen
    toggle is on; takes global resolution + fps + pos/neg prompts. Default 1080p.

Emits two files: a flat graph and a native-subgraph variant (one box per meta stage).
Edit CONFIG, then:  python3 generator/build_workflow.py
"""

import json
import os
import uuid

# --------------------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------------------

CONFIG = {
    # global resolution (drives image lanes + video). LTX needs mult-of-64 -> mult-of-32 stage-1.
    # 1080p default.
    "out_w": 1920, "out_h": 1088,           # output resolution (image lanes + final video)
    "stage1_w": 960, "stage1_h": 544,       # LTX stage-1 = out/2  (edit with out_w/out_h)
    "fps": 24,
    "num_frames": 97,                        # must satisfy (n-1) % 8 == 0

    # per-lane generation toggles (is this frame generated + used by video?)
    "gen_first": True, "gen_mid": False, "gen_last": True,
    # per-lane reference-image toggles (style/composition via img2img)
    "ref_first": False, "ref_mid": False, "ref_last": False,
    # per-lane inpaint toggles (default off)
    "inpaint_first": False, "inpaint_mid": False, "inpaint_last": False,

    # LTX keyframe strengths (stage1, stage2)
    "strength_first": (0.95, 1.0), "strength_mid": (0.5, 0.5), "strength_last": (0.75, 0.8),
    "img2img_denoise": 0.8,                  # strength of reference-image influence
    "seed": 43,

    "ideogram": {
        "diffusion": "ideogram4_fp8_scaled.safetensors",
        "unconditional": "ideogram4_unconditional_fp8_scaled.safetensors",
        "clip": "qwen3vl_8b_fp8_scaled.safetensors", "vae": "flux2-vae.safetensors",
        "steps": 20, "mu": 0.0, "std": 1.75, "cfg": 7,
    },
    "flux_fill": {
        "diffusion": "flux1-fill-dev.safetensors", "vae": "ae.safetensors",
        "clip_l": "clip_l.safetensors", "t5xxl": "t5xxl_fp16.safetensors",
        "guidance": 30.0, "steps": 20, "cfg": 2.0,
    },
    "ltx": {
        "checkpoint": "ltx-2.3-22b-dev.safetensors",
        "distilled_lora": "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
        "spatial_upscaler": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "text_encoder": "comfy_gemma_3_12B_it.safetensors",
    },
}

SIGMAS_S1 = "1.0,0.99375,0.9875,0.98125,0.975,0.909375,0.725,0.421875,0.0"
SIGMAS_S2 = "0.85,0.7250,0.4219,0.0"


# --------------------------------------------------------------------------------------
# Framework
# --------------------------------------------------------------------------------------

# Node types whose single widget is meant to be user-tunable from OUTSIDE a collapsed
# subgraph box (via properties.proxyWidgets on the instance) -> widget key name, verified
# against live /object_info on the user's ComfyUI (0.26.0).
WIDGET_KEY = {
    "PrimitiveBoolean": "value", "PrimitiveInt": "value", "PrimitiveFloat": "value",
    "PrimitiveString": "value", "PrimitiveStringMultiline": "value",
    "CLIPTextEncode": "text", "LoadImage": "image",
}


class Node:
    def __init__(self, nid, ntype, pos, title=None, size=None, tunable=False):
        self.id = nid
        self.type = ntype
        self.title = title
        self.pos = list(pos)
        self.size = size or [300, 120]
        self.flags = {}
        self.order = 0
        self.mode = 0
        self.inputs = []
        self.outputs = []
        self.widgets_values = []
        self.properties = {"Node name for S&R": ntype}
        self.block = "Top"
        self.tunable = tunable

    def add_input(self, name, dtype, widget=None, shape=None):
        e = {"name": name, "type": dtype, "link": None}
        if widget:
            e["widget"] = {"name": widget}
        if shape is not None:
            e["shape"] = shape
        self.inputs.append(e)
        return len(self.inputs) - 1

    def add_output(self, name, dtype):
        self.outputs.append({"name": name, "type": dtype, "links": [], "slot_index": len(self.outputs)})
        return len(self.outputs) - 1

    def _idx(self, lst, key):
        if isinstance(key, int):
            return key
        for i, o in enumerate(lst):
            if o["name"] == key:
                return i
        raise KeyError(f"{key!r} not on {self.type}#{self.id}")

    def to_dict(self):
        d = {"id": self.id, "type": self.type, "pos": self.pos,
             "size": {"0": self.size[0], "1": self.size[1]}, "flags": self.flags,
             "order": self.order, "mode": self.mode, "inputs": self.inputs, "outputs": self.outputs,
             "properties": self.properties, "widgets_values": self.widgets_values}
        if self.title:
            d["title"] = self.title
        return d


class Graph:
    def __init__(self):
        self.nodes = {}
        self._nid = 0
        self.links = []
        self._lid = 0
        self.groups = []
        self._gid = 0
        self._order = 0
        self.block = "Top"

    def node(self, ntype, pos=(0, 0), inputs=None, outputs=None, widgets=None, title=None, size=None,
             tunable=False):
        self._nid += 1
        n = Node(self._nid, ntype, pos, title=title, size=size, tunable=tunable)
        n.block = self.block
        n.order = self._order; self._order += 1
        for s in (inputs or []):
            n.add_input(s[0], s[1], s[2] if len(s) > 2 else None)
        for s in (outputs or []):
            n.add_output(s[0], s[1])
        n.widgets_values = list(widgets or [])
        self.nodes[n.id] = n
        return n

    def link(self, src, src_slot, dst, dst_slot, dtype):
        s = src._idx(src.outputs, src_slot)
        d = dst._idx(dst.inputs, dst_slot)
        self._lid += 1
        lid = self._lid
        src.outputs[s]["links"].append(lid)
        dst.inputs[d]["link"] = lid
        dst.inputs[d]["type"] = dtype
        self.links.append([lid, src.id, s, dst.id, d, dtype])
        return lid

    def to_dict(self):
        return {
            "id": str(uuid.uuid5(uuid.NAMESPACE_URL, "ltx23-3meta")),
            "revision": 0,
            "last_node_id": self._nid, "last_link_id": self._lid,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "links": self.links, "groups": self.groups, "config": {}, "extra": {}, "version": 0.4,
        }


G = Graph()


# --------------------------------------------------------------------------------------
# Top-level: global resolution
# --------------------------------------------------------------------------------------

def build_global():
    G.block = "Top"
    x = -2200
    out_w = G.node("PrimitiveInt", (x, 0), outputs=[("INT", "INT")], widgets=[CONFIG["out_w"]], title="global width")
    out_h = G.node("PrimitiveInt", (x, 120), outputs=[("INT", "INT")], widgets=[CONFIG["out_h"]], title="global height")
    s1_w = G.node("PrimitiveInt", (x, 240), outputs=[("INT", "INT")], widgets=[CONFIG["stage1_w"]], title="video stage1 width")
    s1_h = G.node("PrimitiveInt", (x, 360), outputs=[("INT", "INT")], widgets=[CONFIG["stage1_h"]], title="video stage1 height")
    fps = G.node("PrimitiveFloat", (x, 480), outputs=[("FLOAT", "FLOAT")], widgets=[float(CONFIG["fps"])], title="global fps")

    # Duration: exposed as seconds at the TOP level (was previously only a python CONFIG
    # constant buried inside the Video Generation subgraph -> invisible from outside, the
    # confirmed cause of "missing duration"). A ComfyMathExpression converts seconds+fps
    # into a num_frames value that always satisfies LTX's (n-1)%8==0 latent constraint,
    # using the floor-division operator `//` (an arithmetic operator, safer bet across
    # expression-evaluator backends than calling round()/floor() as named functions) --
    # verified against the user's own working ComfyMathExpression usage in LTS-2.3-1080p.json,
    # whose simpler `a*b+1` formula only stays valid by coincidence when fps is a multiple of 8.
    seconds_default = round(CONFIG["num_frames"] / CONFIG["fps"], 3)
    duration_s = G.node("PrimitiveFloat", (x, 600), outputs=[("FLOAT", "FLOAT")],
                        widgets=[seconds_default], title="video duration (seconds)")
    frames_expr = G.node("ComfyMathExpression", (x, 720), title="duration -> num_frames (8k+1 safe)")
    ia = frames_expr.add_input("values.a", "FLOAT,INT,BOOLEAN")
    ib = frames_expr.add_input("values.b", "FLOAT,INT,BOOLEAN", shape=7)
    frames_expr.add_input("expression", "STRING", widget="expression")
    frames_expr.add_output("FLOAT", "FLOAT")
    frames_expr.add_output("INT", "INT")
    frames_expr.add_output("BOOL", "BOOLEAN")
    frames_expr.widgets_values = ["(a*b-1)//8*8+1"]
    G.link(duration_s, "FLOAT", frames_expr, ia, "FLOAT,INT,BOOLEAN")
    G.link(fps, "FLOAT", frames_expr, ib, "FLOAT,INT,BOOLEAN")

    return {"out_w": out_w, "out_h": out_h, "s1_w": s1_w, "s1_h": s1_h, "fps": fps,
            "duration_s": duration_s, "num_frames": frames_expr}


# --------------------------------------------------------------------------------------
# META 1: Image Generation  (3 optional lanes, shared Ideogram loaders, global resolution)
# --------------------------------------------------------------------------------------

def build_ideogram_loaders():
    c = CONFIG["ideogram"]
    unet = G.node("UNETLoader", (-1500, 0), outputs=[("MODEL", "MODEL")], widgets=[c["diffusion"], "default"])
    unet_neg = G.node("UNETLoader", (-1500, 180), outputs=[("MODEL", "MODEL")], widgets=[c["unconditional"], "default"])
    clip = G.node("CLIPLoader", (-1500, 360), outputs=[("CLIP", "CLIP")], widgets=[c["clip"], "ideogram4", "default"])
    vae = G.node("VAELoader", (-1500, 540), outputs=[("VAE", "VAE")], widgets=[c["vae"]])
    return {"unet": unet, "unet_neg": unet_neg, "clip": clip, "vae": vae}


def build_gen_lane(label, y, loaders, res, ref_default):
    c = CONFIG["ideogram"]
    x = -1000
    pos = G.node("CLIPTextEncode", (x, y), inputs=[("clip", "CLIP")],
                 outputs=[("CONDITIONING", "CONDITIONING")],
                 widgets=[f"{label} frame: describe the shot"], title=f"{label} positive prompt",
                 tunable=True)
    G.link(loaders["clip"], "CLIP", pos, "clip", "CLIP")
    neg = G.node("ConditioningZeroOut", (x, y + 120), inputs=[("conditioning", "CONDITIONING")],
                 outputs=[("CONDITIONING", "CONDITIONING")], title=f"{label} negative (zero)")
    G.link(pos, "CONDITIONING", neg, "conditioning", "CONDITIONING")

    cfg = G.node("CFGOverride", (x, y + 240), inputs=[("model", "MODEL")], outputs=[("MODEL", "MODEL")],
                 widgets=[3, 0.7, 1])
    G.link(loaders["unet"], "MODEL", cfg, "model", "MODEL")
    guider = G.node("DualModelGuider", (x, y + 360),
                    inputs=[("model", "MODEL"), ("positive", "CONDITIONING"),
                            ("model_negative", "MODEL"), ("negative", "CONDITIONING")],
                    outputs=[("GUIDER", "GUIDER")], widgets=[c["cfg"]])
    G.link(cfg, "MODEL", guider, "model", "MODEL")
    G.link(pos, "CONDITIONING", guider, "positive", "CONDITIONING")
    G.link(loaders["unet_neg"], "MODEL", guider, "model_negative", "MODEL")
    G.link(neg, "CONDITIONING", guider, "negative", "CONDITIONING")

    empty_lat = G.node("EmptyFlux2LatentImage", (x, y + 500),
                       inputs=[("width", "INT", "width"), ("height", "INT", "height")],
                       outputs=[("LATENT", "LATENT")], widgets=[CONFIG["out_w"], CONFIG["out_h"], 1])
    G.link(res["out_w"], "INT", empty_lat, "width", "INT")
    G.link(res["out_h"], "INT", empty_lat, "height", "INT")

    # optional reference image -> img2img latent
    ref_img = G.node("LoadImage", (x - 320, y + 500),
                     inputs=[("image", "COMBO", "image"), ("upload", "IMAGEUPLOAD", "upload")],
                     outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")],
                     widgets=["example.png", "image"], title=f"{label} style/composition ref", tunable=True)
    ref_lat = G.node("VAEEncode", (x, y + 640), inputs=[("pixels", "IMAGE"), ("vae", "VAE")],
                     outputs=[("LATENT", "LATENT")])
    G.link(ref_img, "IMAGE", ref_lat, "pixels", "IMAGE")
    G.link(loaders["vae"], "VAE", ref_lat, "vae", "VAE")

    ref_toggle = G.node("PrimitiveBoolean", (x - 320, y + 760), outputs=[("BOOLEAN", "BOOLEAN")],
                        widgets=[ref_default], title=f"{label} ref image ON/OFF", tunable=True)
    lat_sw = G.node("ComfySwitchNode", (x, y + 780),
                    inputs=[("switch", "BOOLEAN", "switch"), ("on_true", "LATENT"), ("on_false", "LATENT")],
                    outputs=[("output", "LATENT")], widgets=[ref_default])
    G.link(ref_toggle, "BOOLEAN", lat_sw, "switch", "BOOLEAN")
    G.link(ref_lat, "LATENT", lat_sw, "on_true", "LATENT")
    G.link(empty_lat, "LATENT", lat_sw, "on_false", "LATENT")

    sig_txt = G.node("Ideogram4Scheduler", (x, y + 920), outputs=[("SIGMAS", "SIGMAS")],
                     widgets=[c["steps"], CONFIG["out_w"], CONFIG["out_h"], c["mu"], c["std"]])
    sig_img = G.node("BasicScheduler", (x + 320, y + 920),
                     inputs=[("model", "MODEL")], outputs=[("SIGMAS", "SIGMAS")],
                     widgets=["normal", c["steps"], CONFIG["img2img_denoise"]])
    G.link(loaders["unet"], "MODEL", sig_img, "model", "MODEL")
    sig_sw = G.node("ComfySwitchNode", (x, y + 1060),
                    inputs=[("switch", "BOOLEAN", "switch"), ("on_true", "SIGMAS"), ("on_false", "SIGMAS")],
                    outputs=[("output", "SIGMAS")], widgets=[ref_default])
    G.link(ref_toggle, "BOOLEAN", sig_sw, "switch", "BOOLEAN")
    G.link(sig_img, "SIGMAS", sig_sw, "on_true", "SIGMAS")
    G.link(sig_txt, "SIGMAS", sig_sw, "on_false", "SIGMAS")

    samp = G.node("KSamplerSelect", (x, y + 1200), outputs=[("SAMPLER", "SAMPLER")], widgets=["euler"])
    noise = G.node("RandomNoise", (x + 320, y + 1200), outputs=[("NOISE", "NOISE")],
                   widgets=[CONFIG["seed"], "randomize"])
    ksamp = G.node("SamplerCustomAdvanced", (x, y + 1340),
                   inputs=[("noise", "NOISE"), ("guider", "GUIDER"), ("sampler", "SAMPLER"),
                           ("sigmas", "SIGMAS"), ("latent_image", "LATENT")],
                   outputs=[("output", "LATENT"), ("denoised_output", "LATENT")])
    G.link(noise, "NOISE", ksamp, "noise", "NOISE")
    G.link(guider, "GUIDER", ksamp, "guider", "GUIDER")
    G.link(samp, "SAMPLER", ksamp, "sampler", "SAMPLER")
    G.link(sig_sw, "output", ksamp, "sigmas", "SIGMAS")
    G.link(lat_sw, "output", ksamp, "latent_image", "LATENT")

    decode = G.node("VAEDecode", (x, y + 1500), inputs=[("samples", "LATENT"), ("vae", "VAE")],
                    outputs=[("IMAGE", "IMAGE")])
    G.link(ksamp, "output", decode, "samples", "LATENT")
    G.link(loaders["vae"], "VAE", decode, "vae", "VAE")
    preview = G.node("PreviewImage", (x, y + 1660), inputs=[("images", "IMAGE")], outputs=[("images", "IMAGE")],
                     title=f"{label} image out")
    G.link(decode, "IMAGE", preview, "images", "IMAGE")
    return preview


def build_image_generation(res):
    G.block = "ImageGen"
    loaders = build_ideogram_loaders()
    first = build_gen_lane("first", 0, loaders, res, CONFIG["ref_first"])
    mid = build_gen_lane("mid", 1800, loaders, res, CONFIG["ref_mid"])
    last = build_gen_lane("last", 3600, loaders, res, CONFIG["ref_last"])
    return {"first": first, "mid": mid, "last": last}


# --------------------------------------------------------------------------------------
# META 2: Inpainting  (3 optional lanes; pass-through when off)
# --------------------------------------------------------------------------------------

def build_flux_loaders():
    c = CONFIG["flux_fill"]
    unet = G.node("UNETLoader", (-200, 0), outputs=[("MODEL", "MODEL")], widgets=[c["diffusion"], "fp8_e4m3fn"])
    clip = G.node("DualCLIPLoader", (-200, 180), outputs=[("CLIP", "CLIP")], widgets=[c["clip_l"], c["t5xxl"], "flux"])
    vae = G.node("VAELoader", (-200, 360), outputs=[("VAE", "VAE")], widgets=[c["vae"]])
    diff = G.node("DifferentialDiffusion", (-200, 540), inputs=[("model", "MODEL")], outputs=[("MODEL", "MODEL")])
    G.link(unet, "MODEL", diff, "model", "MODEL")
    return {"unet": diff, "clip": clip, "vae": vae}


def build_inpaint_lane(label, y, gen_image, flux, inpaint_default):
    c = CONFIG["flux_fill"]
    x = 300
    mask = G.node("LoadImage", (x - 320, y),
                  inputs=[("image", "COMBO", "image"), ("upload", "IMAGEUPLOAD", "upload")],
                  outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")],
                  widgets=["example.png", "image"], title=f"{label} inpaint mask", tunable=True)
    pos = G.node("CLIPTextEncode", (x, y), inputs=[("clip", "CLIP")],
                 outputs=[("CONDITIONING", "CONDITIONING")], widgets=[f"inpaint {label}"],
                 title=f"{label} inpaint prompt", tunable=True)
    G.link(flux["clip"], "CLIP", pos, "clip", "CLIP")
    guide = G.node("FluxGuidance", (x, y + 120), inputs=[("conditioning", "CONDITIONING")],
                   outputs=[("CONDITIONING", "CONDITIONING")], widgets=[c["guidance"]])
    G.link(pos, "CONDITIONING", guide, "conditioning", "CONDITIONING")
    neg = G.node("CLIPTextEncode", (x, y + 240), inputs=[("clip", "CLIP")],
                 outputs=[("CONDITIONING", "CONDITIONING")], widgets=[""],
                 title=f"{label} inpaint negative prompt", tunable=True)
    G.link(flux["clip"], "CLIP", neg, "clip", "CLIP")
    grow = G.node("GrowMask", (x, y + 360), inputs=[("mask", "MASK")], outputs=[("MASK", "MASK")], widgets=[6, 0])
    G.link(mask, "MASK", grow, "mask", "MASK")
    inp = G.node("InpaintModelConditioning", (x, y + 500),
                 inputs=[("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
                         ("vae", "VAE"), ("pixels", "IMAGE"), ("mask", "MASK")],
                 outputs=[("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("latent", "LATENT")],
                 widgets=[False])
    G.link(guide, "CONDITIONING", inp, "positive", "CONDITIONING")
    G.link(neg, "CONDITIONING", inp, "negative", "CONDITIONING")
    G.link(flux["vae"], "VAE", inp, "vae", "VAE")
    G.link(gen_image, "images", inp, "pixels", "IMAGE")
    G.link(grow, "MASK", inp, "mask", "MASK")
    ksamp = G.node("KSampler", (x, y + 680),
                   inputs=[("model", "MODEL"), ("positive", "CONDITIONING"),
                           ("negative", "CONDITIONING"), ("latent_image", "LATENT")],
                   outputs=[("LATENT", "LATENT")],
                   widgets=[CONFIG["seed"], "fixed", c["steps"], c["cfg"], "euler", "normal", 1.0])
    G.link(flux["unet"], "MODEL", ksamp, "model", "MODEL")
    G.link(inp, "positive", ksamp, "positive", "CONDITIONING")
    G.link(inp, "negative", ksamp, "negative", "CONDITIONING")
    G.link(inp, "latent", ksamp, "latent_image", "LATENT")
    dec = G.node("VAEDecode", (x, y + 860), inputs=[("samples", "LATENT"), ("vae", "VAE")],
                 outputs=[("IMAGE", "IMAGE")])
    G.link(ksamp, "LATENT", dec, "samples", "LATENT")
    G.link(flux["vae"], "VAE", dec, "vae", "VAE")

    toggle = G.node("PrimitiveBoolean", (x - 320, y + 860), outputs=[("BOOLEAN", "BOOLEAN")],
                    widgets=[inpaint_default], title=f"{label} inpaint ON/OFF", tunable=True)
    sw = G.node("ComfySwitchNode", (x + 360, y + 860),
                inputs=[("switch", "BOOLEAN", "switch"), ("on_true", "IMAGE"), ("on_false", "IMAGE")],
                outputs=[("output", "IMAGE")], widgets=[inpaint_default], title=f"{label} inpainted image out")
    G.link(toggle, "BOOLEAN", sw, "switch", "BOOLEAN")
    G.link(dec, "IMAGE", sw, "on_true", "IMAGE")
    G.link(gen_image, "images", sw, "on_false", "IMAGE")
    return sw


def build_inpainting(images):
    G.block = "Inpaint"
    flux = build_flux_loaders()
    first = build_inpaint_lane("first", 0, images["first"], flux, CONFIG["inpaint_first"])
    mid = build_inpaint_lane("mid", 1800, images["mid"], flux, CONFIG["inpaint_mid"])
    last = build_inpaint_lane("last", 3600, images["last"], flux, CONFIG["inpaint_last"])
    return {"first": first, "mid": mid, "last": last}


# --------------------------------------------------------------------------------------
# META 3: Video Generation  (LTX-2.3 two-stage, gated keyframes)
# --------------------------------------------------------------------------------------

def build_video(images, res):
    G.block = "Video"
    c = CONFIG["ltx"]
    x = 900
    ckpt = G.node("CheckpointLoaderSimple", (x, 0),
                  outputs=[("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE")], widgets=[c["checkpoint"]])
    lora = G.node("LoraLoaderModelOnly", (x, 200), inputs=[("model", "MODEL")], outputs=[("MODEL", "MODEL")],
                  widgets=[c["distilled_lora"], 0.5])
    G.link(ckpt, "MODEL", lora, "model", "MODEL")
    audio_vae = G.node("LTXVAudioVAELoader", (x, 380), outputs=[("Audio VAE", "VAE")], widgets=[c["checkpoint"]])
    upscaler = G.node("LatentUpscaleModelLoader", (x, 560), outputs=[("LATENT_UPSCALE_MODEL", "LATENT_UPSCALE_MODEL")],
                      widgets=[c["spatial_upscaler"]])
    enc = G.node("LTXAVTextEncoderLoader", (x, 740), outputs=[("CLIP", "CLIP")],
                 widgets=[c["text_encoder"], c["checkpoint"], "default"])

    pos = G.node("CLIPTextEncode", (x, 940), inputs=[("clip", "CLIP")],
                 outputs=[("CONDITIONING", "CONDITIONING")],
                 widgets=["describe the motion/action + audio for the video"], title="video positive prompt",
                 tunable=True)
    G.link(enc, "CLIP", pos, "clip", "CLIP")
    neg = G.node("CLIPTextEncode", (x, 1120), inputs=[("clip", "CLIP")],
                 outputs=[("CONDITIONING", "CONDITIONING")],
                 widgets=["pc game, console game, video game, cartoon, childish, ugly"], title="video negative prompt",
                 tunable=True)
    G.link(enc, "CLIP", neg, "clip", "CLIP")
    cond = G.node("LTXVConditioning", (x, 1300),
                  inputs=[("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("frame_rate", "FLOAT")],
                  outputs=[("positive", "CONDITIONING"), ("negative", "CONDITIONING")], widgets=[])
    G.link(pos, "CONDITIONING", cond, "positive", "CONDITIONING")
    G.link(neg, "CONDITIONING", cond, "negative", "CONDITIONING")
    G.link(res["fps"], "FLOAT", cond, "frame_rate", "FLOAT")

    vid_lat = G.node("EmptyLTXVLatentVideo", (x, 1480),
                     inputs=[("width", "INT", "width"), ("height", "INT", "height"), ("length", "INT", "length")],
                     outputs=[("LATENT", "LATENT")], widgets=[CONFIG["stage1_w"], CONFIG["stage1_h"], CONFIG["num_frames"], 1])
    G.link(res["s1_w"], "INT", vid_lat, "width", "INT")
    G.link(res["s1_h"], "INT", vid_lat, "height", "INT")
    G.link(res["num_frames"], "INT", vid_lat, "length", "INT")
    aud_lat = G.node("LTXVEmptyLatentAudio", (x, 1660),
                     inputs=[("audio_vae", "VAE"), ("frames_number", "INT", "frames_number"),
                             ("frame_rate", "INT", "frame_rate")],
                     outputs=[("Latent", "LATENT")], widgets=[97, 25, 1])
    G.link(audio_vae, "Audio VAE", aud_lat, "audio_vae", "VAE")
    G.link(res["num_frames"], "INT", aud_lat, "frames_number", "INT")
    fps_int = G.node("LTXFloatToInt", (x - 320, 1800), inputs=[("a", "FLOAT")], outputs=[("INT", "INT")])
    G.link(res["fps"], "FLOAT", fps_int, "a", "FLOAT")
    G.link(fps_int, "INT", aud_lat, "frame_rate", "INT")

    # Mid-frame index must track num_frames live (duration is now a UI-tunable, not just a
    # python constant) -> compute it in-graph instead of baking a stale python int.
    mid_idx_expr = G.node("ComfyMathExpression", (x - 320, 1980), title="mid frame index (live, 8k-aligned)")
    mia = mid_idx_expr.add_input("values.a", "FLOAT,INT,BOOLEAN")
    mid_idx_expr.add_input("expression", "STRING", widget="expression")
    mid_idx_expr.add_output("FLOAT", "FLOAT")
    mid_idx_expr.add_output("INT", "INT")
    mid_idx_expr.add_output("BOOL", "BOOLEAN")
    mid_idx_expr.widgets_values = ["((a-1)//2)//8*8"]
    G.link(res["num_frames"], "INT", mid_idx_expr, mia, "FLOAT,INT,BOOLEAN")

    frames = {
        "first": (images["first"], 0, CONFIG["strength_first"][0], CONFIG["strength_first"][1], CONFIG["gen_first"]),
        "mid": (images["mid"], mid_idx_expr, CONFIG["strength_mid"][0], CONFIG["strength_mid"][1], CONFIG["gen_mid"]),
        "last": (images["last"], -1, CONFIG["strength_last"][0], CONFIG["strength_last"][1], CONFIG["gen_last"]),
    }
    # gen toggles live in the Video block (they gate whether a frame's image is pulled into LTX)
    t_first = G.node("PrimitiveBoolean", (x - 320, 0), outputs=[("BOOLEAN", "BOOLEAN")],
                     widgets=[CONFIG["gen_first"]], title="first gen ON/OFF", tunable=True)
    t_mid = G.node("PrimitiveBoolean", (x - 320, 1800 * 0 + 1480), outputs=[("BOOLEAN", "BOOLEAN")],
                   widgets=[CONFIG["gen_mid"]], title="mid gen ON/OFF", tunable=True)
    t_last = G.node("PrimitiveBoolean", (x - 320, 2200), outputs=[("BOOLEAN", "BOOLEAN")],
                    widgets=[CONFIG["gen_last"]], title="last gen ON/OFF", tunable=True)
    toggles = {"first": t_first, "mid": t_mid, "last": t_last}

    ltx = {"vae": ckpt, "lora": lora, "audio_vae": audio_vae, "upscaler": upscaler}
    s1_pos, s1_neg, s1_lat = _guide_stage("stage1", x + 500, 0, cond, vid_lat, frames, toggles, ltx)

    concat1 = G.node("LTXVConcatAVLatent", (x + 1100, 0),
                     inputs=[("video_latent", "LATENT"), ("audio_latent", "LATENT")], outputs=[("latent", "LATENT")])
    G.link(s1_lat, "output", concat1, "video_latent", "LATENT")
    G.link(aud_lat, "Latent", concat1, "audio_latent", "LATENT")
    g1 = G.node("CFGGuider", (x + 1100, 160),
                inputs=[("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING")],
                outputs=[("GUIDER", "GUIDER")], widgets=[1])
    G.link(lora, "MODEL", g1, "model", "MODEL")
    G.link(s1_pos, "output", g1, "positive", "CONDITIONING")
    G.link(s1_neg, "output", g1, "negative", "CONDITIONING")
    samp1 = G.node("KSamplerSelect", (x + 1100, 300), outputs=[("SAMPLER", "SAMPLER")], widgets=["euler_ancestral_cfg_pp"])
    noise1 = G.node("RandomNoise", (x + 1100, 420), outputs=[("NOISE", "NOISE")], widgets=[CONFIG["seed"], "fixed"])
    sig1 = G.node("ManualSigmas", (x + 1100, 540), outputs=[("SIGMAS", "SIGMAS")], widgets=[SIGMAS_S1])
    ks1 = G.node("SamplerCustomAdvanced", (x + 1100, 680),
                 inputs=[("noise", "NOISE"), ("guider", "GUIDER"), ("sampler", "SAMPLER"),
                         ("sigmas", "SIGMAS"), ("latent_image", "LATENT")],
                 outputs=[("output", "LATENT"), ("denoised_output", "LATENT")])
    for src, slot, dst in [(noise1, "NOISE", "noise"), (g1, "GUIDER", "guider"), (samp1, "SAMPLER", "sampler"),
                           (sig1, "SIGMAS", "sigmas"), (concat1, "latent", "latent_image")]:
        G.link(src, slot, ks1, dst, {"noise": "NOISE", "guider": "GUIDER", "sampler": "SAMPLER",
                                     "sigmas": "SIGMAS", "latent_image": "LATENT"}[dst])
    sep1 = G.node("LTXVSeparateAVLatent", (x + 1100, 860), inputs=[("av_latent", "LATENT")],
                  outputs=[("video_latent", "LATENT"), ("audio_latent", "LATENT")])
    G.link(ks1, "output", sep1, "av_latent", "LATENT")

    up = G.node("LTXVLatentUpsampler", (x + 1500, 0),
                inputs=[("samples", "LATENT"), ("upscale_model", "LATENT_UPSCALE_MODEL"), ("vae", "VAE")],
                outputs=[("LATENT", "LATENT")])
    G.link(sep1, "video_latent", up, "samples", "LATENT")
    G.link(upscaler, "LATENT_UPSCALE_MODEL", up, "upscale_model", "LATENT_UPSCALE_MODEL")
    G.link(ckpt, "VAE", up, "vae", "VAE")

    s2_pos, s2_neg, s2_lat = _guide_stage("stage2", x + 1900, 0, cond, up, frames, toggles, ltx)
    concat2 = G.node("LTXVConcatAVLatent", (x + 2400, 0),
                     inputs=[("video_latent", "LATENT"), ("audio_latent", "LATENT")], outputs=[("latent", "LATENT")])
    G.link(s2_lat, "output", concat2, "video_latent", "LATENT")
    G.link(sep1, "audio_latent", concat2, "audio_latent", "LATENT")
    g2 = G.node("CFGGuider", (x + 2400, 160),
                inputs=[("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING")],
                outputs=[("GUIDER", "GUIDER")], widgets=[1])
    G.link(lora, "MODEL", g2, "model", "MODEL")
    G.link(s2_pos, "output", g2, "positive", "CONDITIONING")
    G.link(s2_neg, "output", g2, "negative", "CONDITIONING")
    samp2 = G.node("KSamplerSelect", (x + 2400, 300), outputs=[("SAMPLER", "SAMPLER")], widgets=["euler_cfg_pp"])
    noise2 = G.node("RandomNoise", (x + 2400, 420), outputs=[("NOISE", "NOISE")], widgets=[CONFIG["seed"] + 1, "fixed"])
    sig2 = G.node("ManualSigmas", (x + 2400, 540), outputs=[("SIGMAS", "SIGMAS")], widgets=[SIGMAS_S2])
    ks2 = G.node("SamplerCustomAdvanced", (x + 2400, 680),
                 inputs=[("noise", "NOISE"), ("guider", "GUIDER"), ("sampler", "SAMPLER"),
                         ("sigmas", "SIGMAS"), ("latent_image", "LATENT")],
                 outputs=[("output", "LATENT"), ("denoised_output", "LATENT")])
    for src, slot, dst in [(noise2, "NOISE", "noise"), (g2, "GUIDER", "guider"), (samp2, "SAMPLER", "sampler"),
                           (sig2, "SIGMAS", "sigmas"), (concat2, "latent", "latent_image")]:
        G.link(src, slot, ks2, dst, {"noise": "NOISE", "guider": "GUIDER", "sampler": "SAMPLER",
                                     "sigmas": "SIGMAS", "latent_image": "LATENT"}[dst])
    sep2 = G.node("LTXVSeparateAVLatent", (x + 2400, 860), inputs=[("av_latent", "LATENT")],
                  outputs=[("video_latent", "LATENT"), ("audio_latent", "LATENT")])
    G.link(ks2, "output", sep2, "av_latent", "LATENT")

    vdec = G.node("LTXVTiledVAEDecode", (x + 2900, 0), inputs=[("vae", "VAE"), ("latents", "LATENT")],
                  outputs=[("image", "IMAGE")], widgets=[2, 2, 6, False, "auto", "auto"])
    G.link(ckpt, "VAE", vdec, "vae", "VAE")
    G.link(sep2, "video_latent", vdec, "latents", "LATENT")
    adec = G.node("LTXVAudioVAEDecode", (x + 2900, 200), inputs=[("samples", "LATENT"), ("audio_vae", "VAE")],
                  outputs=[("Audio", "AUDIO")])
    G.link(sep2, "audio_latent", adec, "samples", "LATENT")
    G.link(audio_vae, "Audio VAE", adec, "audio_vae", "VAE")
    cvid = G.node("CreateVideo", (x + 2900, 400),
                  inputs=[("images", "IMAGE"), ("audio", "AUDIO"), ("fps", "FLOAT")],
                  outputs=[("VIDEO", "VIDEO")], widgets=[])
    G.link(vdec, "image", cvid, "images", "IMAGE")
    G.link(adec, "Audio", cvid, "audio", "AUDIO")
    G.link(res["fps"], "FLOAT", cvid, "fps", "FLOAT")
    save = G.node("SaveVideo", (x + 2900, 600), inputs=[("video", "VIDEO")],
                  widgets=["LTX23_3meta", "auto", "auto"])
    G.link(cvid, "VIDEO", save, "video", "VIDEO")


def _guide_stage(stage, x, y0, cond, base_lat, frames, toggles, ltx):
    """Chain 3 gated LTXVAddGuide (first/mid/last). Returns final pos/neg/lat switch nodes."""
    cur_pos, cur_neg, cur_lat = cond, cond, base_lat
    pos_slot, neg_slot = "positive", "negative"
    lat_slot = "LATENT" if base_lat.type in ("EmptyLTXVLatentVideo", "LTXVLatentUpsampler") else "output"
    for name in ("first", "mid", "last"):
        img, idx, s1, s2, _ = frames[name]
        strength = s1 if stage == "stage1" else s2
        idx_literal = idx if isinstance(idx, int) else 0
        guide = G.node("LTXVAddGuide", (x, y0),
                       inputs=[("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
                               ("vae", "VAE"), ("latent", "LATENT"), ("image", "IMAGE")],
                       outputs=[("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("latent", "LATENT")],
                       widgets=[idx_literal, strength], title=f"{stage} guide {name}")
        # frame_idx is always a real convertible input (matches LTXVAddGuide's live schema);
        # only "mid" actually links it (to the live 8k-aligned expression) since first/last
        # are structurally fixed (0 = start, -1 = end, counted from the end by LTX itself).
        fi = guide.add_input("frame_idx", "INT", widget="frame_idx")
        if not isinstance(idx, int):
            G.link(idx, "INT", guide, fi, "INT")
        G.link(cur_pos, pos_slot, guide, "positive", "CONDITIONING")
        G.link(cur_neg, neg_slot, guide, "negative", "CONDITIONING")
        G.link(ltx["vae"], "VAE", guide, "vae", "VAE")
        G.link(cur_lat, lat_slot, guide, "latent", "LATENT")
        G.link(img, "output", guide, "image", "IMAGE")
        swp = G.node("ComfySwitchNode", (x + 360, y0),
                     inputs=[("switch", "BOOLEAN", "switch"), ("on_true", "CONDITIONING"), ("on_false", "CONDITIONING")],
                     outputs=[("output", "CONDITIONING")], widgets=[False])
        swn = G.node("ComfySwitchNode", (x + 360, y0 + 120),
                     inputs=[("switch", "BOOLEAN", "switch"), ("on_true", "CONDITIONING"), ("on_false", "CONDITIONING")],
                     outputs=[("output", "CONDITIONING")], widgets=[False])
        swl = G.node("ComfySwitchNode", (x + 360, y0 + 240),
                     inputs=[("switch", "BOOLEAN", "switch"), ("on_true", "LATENT"), ("on_false", "LATENT")],
                     outputs=[("output", "LATENT")], widgets=[False])
        for sw in (swp, swn, swl):
            G.link(toggles[name], "BOOLEAN", sw, "switch", "BOOLEAN")
        G.link(guide, "positive", swp, "on_true", "CONDITIONING")
        G.link(cur_pos, pos_slot, swp, "on_false", "CONDITIONING")
        G.link(guide, "negative", swn, "on_true", "CONDITIONING")
        G.link(cur_neg, neg_slot, swn, "on_false", "CONDITIONING")
        G.link(guide, "latent", swl, "on_true", "LATENT")
        G.link(cur_lat, lat_slot, swl, "on_false", "LATENT")
        cur_pos, cur_neg, cur_lat = swp, swn, swl
        pos_slot = neg_slot = lat_slot = "output"
        y0 += 380
    return cur_pos, cur_neg, cur_lat


# --------------------------------------------------------------------------------------
# Native subgraph emission (Top stays top-level; other blocks become subgraph boxes)
# --------------------------------------------------------------------------------------

NAMES = {"ImageGen": "Image Generation", "Inpaint": "Inpainting", "Video": "Video Generation"}
ORDER = ["ImageGen", "Inpaint", "Video"]
TOP = "Top"


def emit_subgraphs(G):
    node_block = {n.id: n.block for n in G.nodes.values()}
    blocks = {}
    for n in G.nodes.values():
        blocks.setdefault(n.block, []).append(n)
    sub_blocks = [b for b in blocks if b != TOP]

    buuid = {b: str(uuid.uuid5(uuid.NAMESPACE_URL, "ltx23-3meta/" + b)) for b in sub_blocks}
    defs = {}
    for b in sub_blocks:
        ns = blocks[b]
        defs[b] = {
            "id": buuid[b], "version": 1, "revision": 0, "name": NAMES.get(b, b),
            "category": "LTX23", "description": "",
            "inputNode": {"id": -10, "bounding": [-260, -100, 150, 240]},
            "outputNode": {"id": -20, "bounding": [640, -100, 150, 240]},
            "inputs": [], "outputs": [], "widgets": [], "groups": [],
            "state": {"lastNodeId": max((x.id for x in ns), default=0), "lastLinkId": 0,
                      "lastGroupId": 0, "lastRerouteId": 0},
            "config": {}, "extra": {}, "nodes": [x.to_dict() for x in ns], "links": [],
        }

    instances = {}
    _iid = [10000]
    pos = {"ImageGen": (-400, -300), "Inpaint": (500, -300), "Video": (1400, -300)}
    for b in ORDER:
        if b not in sub_blocks:
            continue
        _iid[0] += 1
        instances[b] = {
            "id": _iid[0], "type": buuid[b], "pos": list(pos.get(b, (0, 0))),
            "size": {"0": 320, "1": 260}, "flags": {}, "order": 0, "mode": 0,
            "inputs": [], "outputs": [],
            # proxyWidgets surfaces widgets from nodes buried INSIDE the collapsed subgraph
            # (toggles, prompts, reference/mask images) directly on the outer box -- without
            # this, every same-block-internal toggle/prompt is unreachable without opening
            # the box, which was the confirmed cause of "missing optionalities" (see TODO.md).
            "properties": {"cnr_id": "comfy-core", "ver": "0.3.43",
                           "proxyWidgets": [[str(n.id), WIDGET_KEY[n.type]]
                                            for n in sorted(blocks[b], key=lambda x: x.order)
                                            if n.tunable]},
            "widgets_values": [],
        }

    out_count = {b: 0 for b in sub_blocks}
    in_count = {b: 0 for b in sub_blocks}
    out_slot, in_slot = {}, {}
    _tlk = [30000]
    _seen = set()
    top_links = []

    # The boundary-segment link inside a subgraph def MUST reuse the ORIGINAL whole-graph
    # link id (lid), not a freshly allocated one: the real internal node's own
    # inputs[...]["link"] / outputs[...]["links"] fields were set to `lid` back when
    # G.link() first ran, and are never rewritten afterwards. Allocating a new id here
    # (as a prior version of this function did) desyncs the node's stored link id from
    # what's actually present in defs[b]["links"], which is exactly the
    # "No link found in parent graph for id [instance:node] slot" error ComfyUI reports
    # when opening the subgraph.
    # Prefer the crossing SOURCE node's title as the human-readable label for an exposed
    # socket (e.g. "first image out") instead of the bare type string -- with 3 parallel
    # lanes all producing the same IMAGE type, an all-"IMAGE" label makes it impossible to
    # visually tell the lanes apart on the collapsed box, which is what raised the "is this
    # a single output fanning out?" question even though the wiring itself was correct.
    node_title = {nid: n.title for nid, n in G.nodes.items() if n.title}

    def expose_output(b, nid, slot, typ, lid):
        key = (b, nid, slot)
        if key not in out_slot:
            ix = out_count[b]; out_count[b] += 1; out_slot[key] = ix
            label = node_title.get(nid, typ)
            defs[b]["outputs"].append({
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{b}-{nid}-{slot}-out")),
                "name": label, "type": typ, "linkIds": [], "localized_name": label, "label": label,
                "pos": [660, 120 + ix * 40]})
        oi = out_slot[key]
        defs[b]["links"].append({"id": lid, "origin_id": nid, "origin_slot": slot,
                                 "target_id": -20, "target_slot": oi, "type": typ})
        defs[b]["outputs"][oi]["linkIds"].append(lid)
        return oi

    def expose_input(b, nid, slot, typ, src_key, lid):
        key = (b,) + src_key
        if key not in in_slot:
            ix = in_count[b]; in_count[b] += 1; in_slot[key] = ix
            label = node_title.get(src_key[1], typ)
            defs[b]["inputs"].append({
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{b}-{src_key[0]}-{src_key[1]}-{src_key[2]}-in")),
                "name": label, "type": typ, "linkIds": [], "localized_name": label, "label": label,
                "pos": [-280, 120 + ix * 40]})
        ii = in_slot[key]
        defs[b]["links"].append({"id": lid, "origin_id": -10, "origin_slot": ii,
                                 "target_id": nid, "target_slot": slot, "type": typ})
        defs[b]["inputs"][ii]["linkIds"].append(lid)
        return ii

    for L in G.links:
        lid, fid, fs, tid, ts, typ = L
        fb, tb = node_block[fid], node_block[tid]
        if fb == tb:
            if fb == TOP:
                top_links.append([lid, fid, fs, tid, ts, typ])  # top -> top
            else:
                defs[fb]["links"].append({"id": lid, "origin_id": fid, "origin_slot": fs,
                                          "target_id": tid, "target_slot": ts, "type": typ})
            continue
        src_key = (fb, fid, fs)
        # from endpoint
        if fb == TOP:
            from_id, from_slot = fid, fs
        else:
            oi = expose_output(fb, fid, fs, typ, lid)
            from_id, from_slot = instances[fb]["id"], oi
        # to endpoint
        if tb == TOP:
            to_id, to_slot = tid, ts
        else:
            ii = expose_input(tb, tid, ts, typ, src_key, lid)
            to_id, to_slot = instances[tb]["id"], ii
        pair = (from_id, from_slot, to_id, to_slot)
        if pair not in _seen:
            _seen.add(pair)
            _tlk[0] += 1
            top_links.append([_tlk[0], from_id, from_slot, to_id, to_slot, typ])

    for b in instances:
        instances[b]["outputs"] = [{"name": o["name"], "type": o["type"], "links": [], "slot_index": i}
                                   for i, o in enumerate(defs[b]["outputs"])]
        instances[b]["inputs"] = [{"name": ip["name"], "type": ip["type"], "link": None}
                                  for ip in defs[b]["inputs"]]
    top_nodes = [n.to_dict() for n in blocks.get(TOP, [])]
    inst_list = [instances[b] for b in ORDER if b in instances]
    all_nodes = top_nodes + inst_list
    # wire top-level links into both endpoints (Top nodes + instances)
    all_map = {n["id"]: n for n in all_nodes}
    for lk in top_links:
        lid, f_id, f_slot, t_id, t_slot, _ = lk
        fn = all_map.get(f_id)
        if fn and f_slot < len(fn["outputs"]):
            fn["outputs"][f_slot]["links"].append(lid)
        tn = all_map.get(t_id)
        if tn and t_slot < len(tn["inputs"]):
            tn["inputs"][t_slot]["link"] = lid
    return {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, "ltx23-3meta-subgraphs")),
        "revision": 0,
        "last_node_id": max((n["id"] for n in all_nodes), default=0),
        "last_link_id": max((l[0] for l in top_links), default=0),
        "nodes": all_nodes, "links": top_links, "groups": [],
        "config": {}, "extra": {}, "version": 0.4,
        "definitions": {"subgraphs": [defs[b] for b in ORDER if b in defs]},
    }


# --------------------------------------------------------------------------------------
# Lint
# --------------------------------------------------------------------------------------

def lint(out):
    errs = []
    nodes = {n["id"]: n for n in out["nodes"]}
    for lk in out["links"]:
        lid, fid, fs, tid, ts, _ = lk
        if fid not in nodes or tid not in nodes:
            errs.append(f"link {lid}: endpoint missing"); continue
        if fs >= len(nodes[fid]["outputs"]):
            errs.append(f"link {lid}: from-slot OOR on {nodes[fid]['type']}#{fid}")
        if ts >= len(nodes[tid]["inputs"]):
            errs.append(f"link {lid}: to-slot OOR on {nodes[tid]['type']}#{tid}"); continue
        if nodes[tid]["inputs"][ts]["link"] != lid:
            errs.append(f"link {lid}: target link-field mismatch")
    if errs:
        print("LINT ERRORS:"); [print("  " + e) for e in errs[:60]]; raise SystemExit(1)
    print(f"lint OK ({len(out['nodes'])} nodes, {len(out['links'])} links)")


def lint_subgraph(sg):
    errs = []
    defids = {d["id"] for d in sg["definitions"]["subgraphs"]}
    nodeids = {n["id"] for n in sg["nodes"]}
    for n in sg["nodes"]:
        if n["type"] not in defids and not n.get("properties", {}).get("Node name for S&R"):
            errs.append(f"top node {n['id']} type {n['type']} unresolved")
    for d in sg["definitions"]["subgraphs"]:
        nn = {x["id"] for x in d["nodes"]}
        for lk in d["links"]:
            for nid, slot in [(lk["origin_id"], lk["origin_slot"]), (lk["target_id"], lk["target_slot"])]:
                if nid == -10:
                    if slot >= len(d["inputs"]):
                        errs.append(f"{d['name']}: link {lk['id']} -10 slot {slot} OOR")
                elif nid == -20:
                    if slot >= len(d["outputs"]):
                        errs.append(f"{d['name']}: link {lk['id']} -20 slot {slot} OOR")
                elif nid not in nn:
                    errs.append(f"{d['name']}: link {lk['id']} node {nid} missing")
        for o in d["outputs"]:
            for lid in o["linkIds"]:
                if not any(l["id"] == lid for l in d["links"]):
                    errs.append(f"{d['name']}: output {o['name']} linkId {lid} missing")
        for ip in d["inputs"]:
            for lid in ip["linkIds"]:
                if not any(l["id"] == lid for l in d["links"]):
                    errs.append(f"{d['name']}: input {ip['name']} linkId {lid} missing")
        # the exact bug class that caused "No link found in parent graph for id
        # [instance:node] slot" in ComfyUI: a node's OWN stored input/output link id(s)
        # must actually exist in this def's "links" list, not just some other id.
        def_link_ids = {lk["id"] for lk in d["links"]}
        for x in d["nodes"]:
            for i, inp in enumerate(x.get("inputs", [])):
                lid = inp.get("link")
                if lid is not None and lid not in def_link_ids:
                    errs.append(f"{d['name']}: node {x['id']} ({x['type']}) input[{i}] "
                                f"{inp.get('name')!r} references link {lid} not in this subgraph's links")
            for i, out in enumerate(x.get("outputs", [])):
                for lid in (out.get("links") or []):
                    if lid not in def_link_ids:
                        errs.append(f"{d['name']}: node {x['id']} ({x['type']}) output[{i}] "
                                    f"{out.get('name')!r} references link {lid} not in this subgraph's links")
    for lk in sg["links"]:
        if lk[1] not in nodeids or lk[3] not in nodeids:
            errs.append(f"top link {lk[0]}: endpoint missing ({lk[1]}->{lk[3]})")
    # proxyWidgets: every [internal_node_id, widget_name] pair must point at a real node
    # inside that instance's subgraph def, with a widget name matching WIDGET_KEY for its type.
    defs_by_id = {d["id"]: d for d in sg["definitions"]["subgraphs"]}
    for n in sg["nodes"]:
        d = defs_by_id.get(n["type"])
        if not d:
            continue
        proxy = n.get("properties", {}).get("proxyWidgets", [])
        inner = {str(x["id"]): x for x in d["nodes"]}
        for nid_str, wname in proxy:
            inner_n = inner.get(nid_str)
            if not inner_n:
                errs.append(f"{d['name']}: proxyWidgets refs missing node {nid_str}")
            elif WIDGET_KEY.get(inner_n["type"]) != wname:
                errs.append(f"{d['name']}: proxyWidgets node {nid_str} ({inner_n['type']}) "
                            f"widget {wname!r} != expected {WIDGET_KEY.get(inner_n['type'])!r}")
    if errs:
        print("SUBGRAPH LINT ERRORS:"); [print("  " + e) for e in errs[:60]]; raise SystemExit(1)
    print(f"subgraph lint OK ({len(sg['definitions']['subgraphs'])} subgraphs, "
          f"{len(sg['nodes'])} top nodes/instances)")


def main():
    res = build_global()
    images = build_image_generation(res)
    inpainted = build_inpainting(images)
    build_video(inpainted, res)

    here = os.path.dirname(os.path.abspath(__file__))
    flat = os.path.normpath(os.path.join(here, "..", "ltx23_keyframe_pipeline.json"))
    with open(flat, "w") as f:
        json.dump(G.to_dict(), f, indent=1)
    print(f"wrote {flat}")
    lint(G.to_dict())

    sg = emit_subgraphs(G)
    sgf = os.path.normpath(os.path.join(here, "..", "ltx23_keyframe_pipeline_subgraphs.json"))
    with open(sgf, "w") as f:
        json.dump(sg, f, indent=1)
    print(f"wrote {sgf}")
    lint_subgraph(sg)


if __name__ == "__main__":
    main()
