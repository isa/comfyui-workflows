#!/usr/bin/env python3
"""
Extract 4 standalone single-purpose workflows from the 3-meta keyframe pipeline.

All 4 are native subgraphs: one collapsed box each, with everything (loaders, guider,
schedulers, sampler, decode, crop) hidden inside. Only a handful of controls are proxied
onto each box; image-seed LoadImage nodes stay REAL external nodes next to the box rather
than proxied widgets (a proxied LoadImage loses its preview thumbnail for already-uploaded
files, breaking mask-editor access -- see emit_subgraph_with_image_seed(s)'s docstrings).

  1. Ideogram-Image-Gen-1080p  — t2i, pos/neg, target res, optional img2img "image seed"
                                 (external LoadImage + ON/OFF toggle proxied on the box).
  2. Flux-Inpaint-1080p        — Flux Fill inpaint, pos/neg, denoise, target res,
                                 maskable image seed (external, mandatory).
  3. Flux-Outpaint-1080p       — Flux Fill outpaint, pos/neg, target res, image seed
                                 (external, mandatory, fit-scaled + padded, never stretched).
  4. LTX-2.3-Video-Gen-1080p   — LTX-2.3 two-stage + audio, pos/neg, width/height, fps,
                                 duration, 3 optional keyframe seeds (external LoadImage +
                                 ON/OFF toggle each) proxied on the box, saves video.

1080p mechanism (uniform): generate at model-friendly 1920x1088 (LTX needs mult-of-64;
keeps Flux latent dims valid) then ImageCrop 4px off top+bottom -> exactly 1920x1080
before Save. Guarantees "absolute 1080p" saved files.

Reuses the verified Node/Graph/lint framework + CONFIG node schemas from build_workflow.py.
Run:  python3 generator/build_extracted_workflows.py
"""

import json
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_workflow import (  # noqa: E402
    Node, Graph, CONFIG, SIGMAS_S1, SIGMAS_S2, lint, WIDGET_KEY, lint_subgraph,
)

# SaveImage.filename_prefix wasn't previously a proxyable widget (the 3-meta pipeline never
# exposed a Save node's filename outward) -- register it so a subgraph instance can surface
# "save filename" as one of its outward widgets.
WIDGET_KEY["SaveImage"] = "filename_prefix"
# Same reasoning: expose Flux Inpaint's denoise so users can trade off "follow the prompt
# strongly" (near 1.0, discards original masked content) vs "preserve original structure,
# shift color/detail only" (lower values) -- see flux_fill_body()'s docstring.
WIDGET_KEY["KSampler"] = "denoise"
# Same reasoning: expose Video Gen's keyframe guide strength. LTXVAddGuide's strength
# widget supports up to 10.0 (confirmed via core node schema, comfy_extras/nodes_lt.py),
# not just 0-1 -- at LTX-2.3's distilled low-CFG sampling, strength is the SOLE mechanism
# enforcing keyframe adherence, and the conservative <=1.0 defaults inherited from the
# original pipeline weren't visibly "locking" the first/last frame at all (user-reported).
WIDGET_KEY["LTXVAddGuide"] = "strength"
# First-frame conditioning switched from LTXVAddGuide to LTXVImgToVideoConditionOnly --
# matches the official Lightricks two-stage reference template exactly and avoids a known,
# still-open bug specific to LTXVAddGuide with combined first+last frame_idx (GH
# Comfy-Org/ComfyUI#12832, "last 1 second very bad" -- exact match for what we hit live).
# Same reasoning as above: expose its own strength widget.
WIDGET_KEY["LTXVImgToVideoConditionOnly"] = "strength"
# Custom node (see generator/custom_nodes/ltx23_helpers) that bundles LoadImage + an
# enable checkbox into ONE node, so each keyframe lane is a single input node. If it's
# ever proxied, its exposeable widget is the enable checkbox.
WIDGET_KEY["LoadImageWithEnable"] = "enable"
# SDXL-Flux outpaint loaders -- exposed as pickers on the box so the user can select the
# SDXL checkpoint / ControlNet Union promax / 4x upscale model they downloaded.
WIDGET_KEY["CheckpointLoaderSimple"] = "ckpt_name"
WIDGET_KEY["ControlNetLoader"] = "control_net_name"

# Multi-GPU model residency (user has 4x A100 40GB, confirmed ComfyUI-MultiGPU
# pollockjj/city96 lineage installed). Each workflow's loaders are pinned to a dedicated
# GPU via that pack's *MultiGPU node variants (UNETLoaderMultiGPU, VAELoaderMultiGPU,
# CLIPLoaderMultiGPU, DualCLIPLoaderMultiGPU, CheckpointLoaderSimpleMultiGPU -- confirmed
# these exist via the pack's actual __init__.py/wrappers.py source, not guessed). This
# does NOT make a single generation's internal steps run concurrently -- ComfyUI executes
# one queued workflow's nodes sequentially regardless of device. What it DOES do: each
# workflow's models stay resident on their OWN GPU instead of competing for space on one
# shared default device, eliminating the repeated evict/reload swap cost, and letting
# multiple DIFFERENT workflows (e.g. an Inpaint + an Outpaint job) run truly concurrently
# without fighting over the same VRAM.
GPU_IMAGEGEN = "cuda:0"
GPU_INPAINT = "cuda:1"
GPU_OUTPAINT = "cuda:2"
GPU_VIDEO = "cuda:3"

C_ID = CONFIG["ideogram"]
C_FF = CONFIG["flux_fill"]
C_LTX = CONFIG["ltx"]
# SDXL+Flux multi-stage outpaint (see build_sdxl_flux_outpaint). The model filenames are
# DEFAULTS -- the user must download them and may rename; both are exposed as tunable
# pickers on the collapsed box. Canvas runs at the full 1920x1088 (cropped to 1080p out) --
# NOT a smaller SDXL-friendly size -- because shrinking a high-res source down to ~576px
# then upscaling produced weak, flat outpainted borders ("frame" artifact). Working at full
# 1080p keeps the source at native detail and gives SDXL/Flux the resolution to extend well.
C_SDXL = {
    "checkpoint": "juggernautXL_v9.safetensors",          # any SDXL 1.0 checkpoint
    "controlnet": "diffusion_pytorch_model_promax.safetensors",  # xinsir/ControlNet-Union-SDXL promax (v2)
    "steps": 25, "cfg": 4.5, "sampler": "dpmpp_2m", "scheduler": "karras",
    "grow_mask": 30,                                       # VAEEncodeForInpaint grow_mask_by (seam blend zone)
}
OUT_W, OUT_H = CONFIG["out_w"], CONFIG["out_h"]          # 1920, 1088 (generate res)
S1_W, S1_H = CONFIG["stage1_w"], CONFIG["stage1_h"]      # 960, 544  (LTX stage-1 = out/2)
SEED = CONFIG["seed"]


# --------------------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------------------

def math_expr(G, pos, expr, a_src, a_slot, title, b_src=None, b_slot=None):
    """A ComfyMathExpression node (values.a[, values.b], expression). Returns the node."""
    n = G.node("ComfyMathExpression", pos, title=title)
    ia = n.add_input("values.a", "FLOAT,INT,BOOLEAN")
    ib = n.add_input("values.b", "FLOAT,INT,BOOLEAN", shape=7)
    n.add_input("expression", "STRING", widget="expression")
    n.add_output("FLOAT", "FLOAT")
    n.add_output("INT", "INT")
    n.add_output("BOOL", "BOOLEAN")
    n.widgets_values = [expr]
    G.link(a_src, a_slot, n, ia, "FLOAT,INT,BOOLEAN")
    if b_src is not None:
        G.link(b_src, b_slot, n, ib, "FLOAT,INT,BOOLEAN")
    return n


def crop_to_1080(G, tw, th, src, src_slot, pos):
    """ImageCrop that removes 4px off top and bottom -> target_w x (target_h-8).

    width tracks `target width`; height = `target height` - 8 (computed live). x=0, y=4.
    """
    ch = math_expr(G, (pos[0], pos[1] - 140), "a-8", th, "INT",
                   title="crop height = target_h - 8")
    crop = G.node("ImageCrop", pos,
                  inputs=[("image", "IMAGE"), ("width", "INT", "width"), ("height", "INT", "height")],
                  outputs=[("IMAGE", "IMAGE")], widgets=[OUT_W, OUT_H - 8, 0, 4],
                  title="crop to 1080p (4px top/bottom)")
    G.link(tw, "INT", crop, "width", "INT")
    G.link(ch, "INT", crop, "height", "INT")
    G.link(src, src_slot, crop, "image", "IMAGE")
    return crop


def res_primitives(G, x=-2200, y=0):
    tw = G.node("PrimitiveInt", (x, y), outputs=[("INT", "INT")], widgets=[OUT_W],
                title="target width", tunable=True)
    th = G.node("PrimitiveInt", (x, y + 120), outputs=[("INT", "INT")], widgets=[OUT_H],
                title="target height", tunable=True)
    return tw, th


def flux_fill_loaders(G, x=-1500, device=None):
    """device: a ComfyUI-MultiGPU device string (e.g. "cuda:1") to pin these loaders'
    models to a dedicated GPU. Uses the pack's *MultiGPU node variants -- confirmed real
    via its actual source (UNETLoaderMultiGPU/VAELoaderMultiGPU/DualCLIPLoaderMultiGPU
    each add a trailing "device" widget onto the vanilla loader). None keeps the vanilla
    node types (whatever device ComfyUI's own default resolves to)."""
    unet_type = "UNETLoaderMultiGPU" if device else "UNETLoader"
    clip_type = "DualCLIPLoaderMultiGPU" if device else "DualCLIPLoader"
    vae_type = "VAELoaderMultiGPU" if device else "VAELoader"
    unet_widgets = [C_FF["diffusion"], "fp8_e4m3fn"] + ([device] if device else [])
    clip_widgets = [C_FF["clip_l"], C_FF["t5xxl"], "flux"] + ([device] if device else [])
    vae_widgets = [C_FF["vae"]] + ([device] if device else [])

    unet = G.node(unet_type, (x, 0), outputs=[("MODEL", "MODEL")], widgets=unet_widgets)
    clip = G.node(clip_type, (x, 180), outputs=[("CLIP", "CLIP")], widgets=clip_widgets)
    vae = G.node(vae_type, (x, 360), outputs=[("VAE", "VAE")], widgets=vae_widgets)
    diff = G.node("DifferentialDiffusion", (x, 540), inputs=[("model", "MODEL")],
                  outputs=[("MODEL", "MODEL")])
    G.link(unet, "MODEL", diff, "model", "MODEL")
    return {"unet": diff, "clip": clip, "vae": vae}


def flux_fill_body(G, flux, tw, th, pixels_src, pixels_slot, mask_src, mask_slot, x=-300,
                    denoise=1.0, seed_control="fixed"):
    """positive/negative prompts + FluxGuidance + InpaintModelConditioning + KSampler + decode.
    Returns (VAEDecode node, positive CLIPTextEncode node, negative CLIPTextEncode node,
    KSampler node). denoise defaults to 1.0 (full regen -- correct when the masked region
    has no original content to preserve, e.g. outpaint's padded border). For inpaint, pass
    a lower value: at denoise=1.0 the masked pixels are pure noise going in, so the model
    has zero anchor on the ORIGINAL shape/structure under the mask and is free to invent a
    different one guided only by the prompt + surrounding context -- confirmed by a live
    test where "make his turban yellow, do NOT change the outlook" still changed the
    turban's shape, because full denoise discards the original silhouette entirely.

    seed_control="fixed" (default) reproduces the same result for the same inputs --
    useful once you've found a good edit. Outpaint passes "randomize" instead: a large
    portrait->landscape outpaint can leave 60%+ of the canvas as newly generated content,
    and Flux Fill can default to a "safe" hallucination (e.g. duplicating the one
    recognizable subject instead of inventing new scenery) for a given seed -- confirmed
    live (a garden outpaint tripled the person instead of extending the background).
    Randomizing means a re-run tries a fresh seed instead of reproducing the exact same
    result every time."""
    pos = G.node("CLIPTextEncode", (x, 0), inputs=[("clip", "CLIP")],
                 outputs=[("CONDITIONING", "CONDITIONING")], widgets=[""],
                 title="positive prompt", tunable=True)
    G.link(flux["clip"], "CLIP", pos, "clip", "CLIP")
    guide = G.node("FluxGuidance", (x, 120), inputs=[("conditioning", "CONDITIONING")],
                   outputs=[("CONDITIONING", "CONDITIONING")], widgets=[C_FF["guidance"]])
    G.link(pos, "CONDITIONING", guide, "conditioning", "CONDITIONING")
    neg = G.node("CLIPTextEncode", (x, 240), inputs=[("clip", "CLIP")],
                 outputs=[("CONDITIONING", "CONDITIONING")], widgets=[""],
                 title="negative prompt", tunable=True)
    G.link(flux["clip"], "CLIP", neg, "clip", "CLIP")
    inp = G.node("InpaintModelConditioning", (x, 380),
                 inputs=[("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
                         ("vae", "VAE"), ("pixels", "IMAGE"), ("mask", "MASK")],
                 outputs=[("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
                          ("latent", "LATENT")], widgets=[False])
    G.link(guide, "CONDITIONING", inp, "positive", "CONDITIONING")
    G.link(neg, "CONDITIONING", inp, "negative", "CONDITIONING")
    G.link(flux["vae"], "VAE", inp, "vae", "VAE")
    G.link(pixels_src, pixels_slot, inp, "pixels", "IMAGE")
    G.link(mask_src, mask_slot, inp, "mask", "MASK")
    ksamp = G.node("KSampler", (x, 560),
                   inputs=[("model", "MODEL"), ("positive", "CONDITIONING"),
                           ("negative", "CONDITIONING"), ("latent_image", "LATENT")],
                   outputs=[("LATENT", "LATENT")],
                   widgets=[SEED, seed_control, C_FF["steps"], C_FF["cfg"], "euler", "normal", denoise],
                   tunable=True)
    G.link(flux["unet"], "MODEL", ksamp, "model", "MODEL")
    G.link(inp, "positive", ksamp, "positive", "CONDITIONING")
    G.link(inp, "negative", ksamp, "negative", "CONDITIONING")
    G.link(inp, "latent", ksamp, "latent_image", "LATENT")
    dec = G.node("VAEDecode", (x, 740), inputs=[("samples", "LATENT"), ("vae", "VAE")],
                 outputs=[("IMAGE", "IMAGE")])
    G.link(ksamp, "LATENT", dec, "samples", "LATENT")
    G.link(flux["vae"], "VAE", dec, "vae", "VAE")
    return dec, pos, neg, ksamp


def dump(G, name):
    here = os.path.dirname(os.path.abspath(__file__))
    out = G.to_dict()
    out["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, "ltx23-extracted/" + name))
    out["floatingLinks"] = []
    path = os.path.normpath(os.path.join(here, "..", name + ".json"))
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {path}")
    lint(out)


def dump_subgraph(sg, name):
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.normpath(os.path.join(here, "..", name + ".json"))
    with open(path, "w") as f:
        json.dump(sg, f, indent=1)
    print(f"wrote {path}")
    lint_subgraph(sg)


def emit_subgraph_with_image_seed(G, name, proxy_nodes, ext_node, wires, category="LTX23"):
    """Like emit_single_subgraph, but `ext_node` (a LoadImage) stays a REAL, un-proxied
    top-level node next to the collapsed box, wired in via a genuine exposed subgraph
    input socket -- not a widget proxy.

    Why: proxyWidgets only mirrors a LoadImage's plain filename dropdown onto the outer
    box; it does not carry over the preview-thumbnail + right-click "Open in MaskEditor"
    wiring the frontend attaches to a real LoadImage instance. Uploading a fresh file
    works (the upload flow sets the preview directly); picking an already-uploaded file
    from the mirrored dropdown never populates a preview, so there's nothing to
    right-click. A genuine, unproxied LoadImage node doesn't have this problem.

    wires: list of (ext_output_name, internal_node, internal_slot_key) tuples -- each
    describes one of ext_node's outputs (e.g. "IMAGE", "MASK") feeding one internal
    node's input. Multiple wires sharing the same ext_output_name collapse onto ONE
    exposed input socket with several internal linkIds (fan-out), matching the
    source-keyed exposure rule the 3-meta pipeline's own subgraph emitter uses.
    """
    ext_out_index = {o["name"]: i for i, o in enumerate(ext_node.outputs)}
    ext_out_type = {o["name"]: o["type"] for o in ext_node.outputs}

    lid = [G._lid]
    def_links = [{"id": l, "origin_id": fid, "origin_slot": fs, "target_id": tid,
                  "target_slot": ts, "type": typ} for l, fid, fs, tid, ts, typ in G.links]

    input_defs = []
    input_slot_ix = {}
    for ext_slot, internal_node, internal_slot_key in wires:
        if ext_slot not in input_slot_ix:
            ix = len(input_defs)
            input_slot_ix[ext_slot] = ix
            input_defs.append({
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{name}-extin-{ext_slot}")),
                "name": ext_slot, "type": ext_out_type[ext_slot], "linkIds": [],
                "localized_name": ext_slot, "label": ext_slot, "pos": [-280, 120 + ix * 40],
            })
        ix = input_slot_ix[ext_slot]
        lid[0] += 1
        typ = ext_out_type[ext_slot]
        islot = internal_node._idx(internal_node.inputs, internal_slot_key)
        internal_node.inputs[islot]["link"] = lid[0]
        def_links.append({"id": lid[0], "origin_id": -10, "origin_slot": ix,
                          "target_id": internal_node.id, "target_slot": islot, "type": typ})
        input_defs[ix]["linkIds"].append(lid[0])

    def_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "ltx23-extracted-sub/" + name))
    subdef = {
        "id": def_id, "version": 1, "revision": 0, "name": name, "category": category,
        "description": "", "inputNode": {"id": -10, "bounding": [-260, -100, 150, 240]},
        "outputNode": {"id": -20, "bounding": [640, -100, 150, 240]},
        "inputs": input_defs, "outputs": [], "widgets": [], "groups": [],
        "state": {"lastNodeId": max((n.id for n in G.nodes.values()), default=0),
                  "lastLinkId": 0, "lastGroupId": 0, "lastRerouteId": 0},
        "config": {}, "extra": {},
        "nodes": [n.to_dict() for n in G.nodes.values()], "links": def_links,
    }

    instance_id = 9001
    instance = {
        "id": instance_id, "type": def_id, "pos": [340, 0], "size": {"0": 320, "1": 400},
        "flags": {}, "order": 1, "mode": 0,
        "inputs": [{"name": s, "type": ext_out_type[s], "link": None} for s in input_slot_ix],
        "outputs": [],
        "properties": {"cnr_id": "comfy-core", "ver": "0.3.43",
                       "proxyWidgets": [[str(n.id), WIDGET_KEY[n.type]] for n in proxy_nodes]},
        "widgets_values": [],
    }

    ext_dict = ext_node.to_dict()
    ext_dict["order"] = 0
    top_links = []
    top_lid = 0
    for s, ix in input_slot_ix.items():
        top_lid += 1
        eo = ext_out_index[s]
        typ = ext_out_type[s]
        top_links.append([top_lid, ext_dict["id"], eo, instance_id, ix, typ])
        ext_dict["outputs"][eo]["links"].append(top_lid)
        instance["inputs"][ix]["link"] = top_lid

    return {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, "ltx23-extracted-sub-top/" + name)),
        "revision": 0, "last_node_id": instance_id, "last_link_id": top_lid,
        "nodes": [ext_dict, instance], "links": top_links, "groups": [], "config": {},
        "extra": {}, "version": 0.4, "definitions": {"subgraphs": [subdef]}, "floatingLinks": [],
    }


GROUP_COLOR = "#3f789e"


def emit_subgraph_with_image_seeds(G, name, proxy_nodes, ext_nodes, wires, category="LTX23"):
    """Like emit_subgraph_with_image_seed, but for MULTIPLE real external LoadImage nodes
    (e.g. Video Gen's first/mid/last keyframe seeds) instead of just one.

    ext_nodes: list of Node objects, each built in a SHARED throwaway Graph() (so their
    ids are unique 1..N -- three separate `Graph()` calls would each start at id=1 and
    collide once placed together in the same top-level `nodes` list).

    wires: list of (ext_node, ext_output_name, internal_node, internal_slot_key) --
    exposure is keyed by (ext_node.id, ext_output_name), so multiple wires from the SAME
    ext_node+output fan onto ONE shared input socket (e.g. a keyframe feeding both the
    stage1 and stage2 LTXVAddGuide), while different ext_nodes always get distinct
    sockets even if they share an output name (three LoadImage.IMAGE outputs must NOT
    collapse onto a single "IMAGE" socket).
    """
    out_index = {n.id: {o["name"]: i for i, o in enumerate(n.outputs)} for n in ext_nodes}
    out_type = {n.id: {o["name"]: o["type"] for o in n.outputs} for n in ext_nodes}

    lid = [G._lid]
    def_links = [{"id": l, "origin_id": fid, "origin_slot": fs, "target_id": tid,
                  "target_slot": ts, "type": typ} for l, fid, fs, tid, ts, typ in G.links]

    input_defs = []
    input_slot_ix = {}
    for ext_node, ext_slot, internal_node, internal_slot_key in wires:
        key = (ext_node.id, ext_slot)
        if key not in input_slot_ix:
            ix = len(input_defs)
            input_slot_ix[key] = ix
            # Label: node title for the primary (IMAGE) socket; for secondary outputs
            # (e.g. ENABLE) append the output name so a multi-output node's sockets read
            # distinctly -- e.g. "first frame" (IMAGE) vs "first frame · enable" (BOOLEAN).
            base = ext_node.title or ext_slot
            label = base if out_type[ext_node.id][ext_slot] == "IMAGE" else f"{base} · {ext_slot.lower()}"
            input_defs.append({
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{name}-extin-{ext_node.id}-{ext_slot}")),
                "name": label, "type": out_type[ext_node.id][ext_slot], "linkIds": [],
                "localized_name": label, "label": label, "pos": [-280, 120 + ix * 40],
            })
        ix = input_slot_ix[key]
        lid[0] += 1
        typ = out_type[ext_node.id][ext_slot]
        islot = internal_node._idx(internal_node.inputs, internal_slot_key)
        internal_node.inputs[islot]["link"] = lid[0]
        def_links.append({"id": lid[0], "origin_id": -10, "origin_slot": ix,
                          "target_id": internal_node.id, "target_slot": islot, "type": typ})
        input_defs[ix]["linkIds"].append(lid[0])

    def_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "ltx23-extracted-sub/" + name))
    subdef = {
        "id": def_id, "version": 1, "revision": 0, "name": name, "category": category,
        "description": "", "inputNode": {"id": -10, "bounding": [-260, -100, 150, 240]},
        "outputNode": {"id": -20, "bounding": [640, -100, 150, 240]},
        "inputs": input_defs, "outputs": [], "widgets": [], "groups": [],
        "state": {"lastNodeId": max((n.id for n in G.nodes.values()), default=0),
                  "lastLinkId": 0, "lastGroupId": 0, "lastRerouteId": 0},
        "config": {}, "extra": {},
        "nodes": [n.to_dict() for n in G.nodes.values()], "links": def_links,
    }

    instance_id = 9001
    instance = {
        "id": instance_id, "type": def_id, "pos": [340 * (len(ext_nodes) + 1), 0],
        "size": {"0": 320, "1": 400}, "flags": {}, "order": len(ext_nodes), "mode": 0,
        "inputs": [{"name": d["name"], "type": d["type"], "link": None} for d in input_defs],
        "outputs": [],
        "properties": {"cnr_id": "comfy-core", "ver": "0.3.43",
                       "proxyWidgets": [[str(n.id), WIDGET_KEY[n.type]] for n in proxy_nodes]},
        "widgets_values": [],
    }

    ext_dicts = []
    for i, ext_node in enumerate(ext_nodes):
        d = ext_node.to_dict()
        d["order"] = i
        d["pos"] = [340 * i, 0]
        ext_dicts.append(d)
    ext_by_id = {d["id"]: d for d in ext_dicts}

    top_links = []
    top_lid = 0
    for (ext_id, ext_slot), ix in input_slot_ix.items():
        top_lid += 1
        eo = out_index[ext_id][ext_slot]
        typ = out_type[ext_id][ext_slot]
        top_links.append([top_lid, ext_id, eo, instance_id, ix, typ])
        ext_by_id[ext_id]["outputs"][eo]["links"].append(top_lid)
        instance["inputs"][ix]["link"] = top_lid

    return {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, "ltx23-extracted-sub-top/" + name)),
        "revision": 0, "last_node_id": instance_id, "last_link_id": top_lid,
        "nodes": ext_dicts + [instance], "links": top_links, "groups": [], "config": {},
        "extra": {}, "version": 0.4, "definitions": {"subgraphs": [subdef]}, "floatingLinks": [],
    }


# --------------------------------------------------------------------------------------
# 1. Ideogram-Image-Gen-1080p  (native subgraph: 1 box, 4 proxied control groups)
# --------------------------------------------------------------------------------------

def build_ideogram():
    G = Graph()
    tw, th = res_primitives(G)

    # Pinned to a dedicated GPU via ComfyUI-MultiGPU's *MultiGPU loader variants -- see
    # GPU_IMAGEGEN's definition for why (model residency, not per-node parallel compute).
    unet = G.node("UNETLoaderMultiGPU", (-1500, 0), outputs=[("MODEL", "MODEL")],
                  widgets=[C_ID["diffusion"], "default", GPU_IMAGEGEN])
    unet_neg = G.node("UNETLoaderMultiGPU", (-1500, 180), outputs=[("MODEL", "MODEL")],
                      widgets=[C_ID["unconditional"], "default", GPU_IMAGEGEN])
    clip = G.node("CLIPLoaderMultiGPU", (-1500, 360), outputs=[("CLIP", "CLIP")],
                  widgets=[C_ID["clip"], "ideogram4", GPU_IMAGEGEN])
    vae = G.node("VAELoaderMultiGPU", (-1500, 540), outputs=[("VAE", "VAE")],
                 widgets=[C_ID["vae"], GPU_IMAGEGEN])

    pos = G.node("CLIPTextEncode", (-1000, 0), inputs=[("clip", "CLIP")],
                 outputs=[("CONDITIONING", "CONDITIONING")],
                 widgets=["describe the image you want"], title="positive prompt", tunable=True)
    G.link(clip, "CLIP", pos, "clip", "CLIP")
    neg = G.node("CLIPTextEncode", (-1000, 120), inputs=[("clip", "CLIP")],
                 outputs=[("CONDITIONING", "CONDITIONING")], widgets=[""],
                 title="negative prompt", tunable=True)
    G.link(clip, "CLIP", neg, "clip", "CLIP")

    cfg = G.node("CFGOverride", (-1000, 240), inputs=[("model", "MODEL")],
                 outputs=[("MODEL", "MODEL")], widgets=[3, 0.7, 1])
    G.link(unet, "MODEL", cfg, "model", "MODEL")
    guider = G.node("DualModelGuider", (-1000, 360),
                    inputs=[("model", "MODEL"), ("positive", "CONDITIONING"),
                            ("model_negative", "MODEL"), ("negative", "CONDITIONING")],
                    outputs=[("GUIDER", "GUIDER")], widgets=[C_ID["cfg"]])
    G.link(cfg, "MODEL", guider, "model", "MODEL")
    G.link(pos, "CONDITIONING", guider, "positive", "CONDITIONING")
    G.link(unet_neg, "MODEL", guider, "model_negative", "MODEL")
    G.link(neg, "CONDITIONING", guider, "negative", "CONDITIONING")

    empty_lat = G.node("EmptyFlux2LatentImage", (-1000, 500),
                       inputs=[("width", "INT", "width"), ("height", "INT", "height")],
                       outputs=[("LATENT", "LATENT")], widgets=[OUT_W, OUT_H, 1])
    G.link(tw, "INT", empty_lat, "width", "INT")
    G.link(th, "INT", empty_lat, "height", "INT")

    # Optional "image seed" = img2img reference. ONE node per lane: LoadImageWithEnable
    # (custom node, generator/custom_nodes/ltx23_helpers) bundles the image picker + an
    # enable checkbox, inheriting LoadImage's upload button / preview thumbnail /
    # mask-editor (the UI attaches to the image_upload widget metadata, not the LoadImage
    # type -- verified via PainterNode precedent). Its ENABLE output fans to the same
    # Switch consumers the separate toggle used to. Enable defaults OFF -- nothing happens
    # with the reference unless explicitly turned on. (Replaces the earlier 2-node
    # LoadImage + PrimitiveBoolean pair: no stock/installed node combines an image picker
    # with a boolean toggle, so a small custom node is the only literal single-node way.)
    EG_ref = Graph()
    ref_img = EG_ref.node("LoadImageWithEnable", (0, 0),
                          inputs=[("image", "COMBO", "image"), ("enable", "BOOLEAN", "enable"),
                                  ("upload", "IMAGEUPLOAD", "upload")],
                          outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK"), ("ENABLE", "BOOLEAN")],
                          widgets=["example.png", False, "image"], title="image seed (optional img2img ref)")

    ref_lat = G.node("VAEEncode", (-1000, 640), inputs=[("pixels", "IMAGE"), ("vae", "VAE")],
                     outputs=[("LATENT", "LATENT")])
    G.link(vae, "VAE", ref_lat, "vae", "VAE")
    lat_sw = G.node("ComfySwitchNode", (-1000, 780),
                    inputs=[("switch", "BOOLEAN", "switch"), ("on_true", "LATENT"), ("on_false", "LATENT")],
                    outputs=[("output", "LATENT")], widgets=[False])
    G.link(ref_lat, "LATENT", lat_sw, "on_true", "LATENT")
    G.link(empty_lat, "LATENT", lat_sw, "on_false", "LATENT")

    sig_txt = G.node("Ideogram4Scheduler", (-1000, 920), outputs=[("SIGMAS", "SIGMAS")],
                     widgets=[C_ID["steps"], OUT_W, OUT_H, C_ID["mu"], C_ID["std"]])
    sig_img = G.node("BasicScheduler", (-680, 920), inputs=[("model", "MODEL")],
                     outputs=[("SIGMAS", "SIGMAS")], widgets=["normal", C_ID["steps"], CONFIG["img2img_denoise"]])
    G.link(unet, "MODEL", sig_img, "model", "MODEL")
    sig_sw = G.node("ComfySwitchNode", (-1000, 1060),
                    inputs=[("switch", "BOOLEAN", "switch"), ("on_true", "SIGMAS"), ("on_false", "SIGMAS")],
                    outputs=[("output", "SIGMAS")], widgets=[False])
    G.link(sig_img, "SIGMAS", sig_sw, "on_true", "SIGMAS")
    G.link(sig_txt, "SIGMAS", sig_sw, "on_false", "SIGMAS")

    samp = G.node("KSamplerSelect", (-1000, 1200), outputs=[("SAMPLER", "SAMPLER")], widgets=["euler"])
    noise = G.node("RandomNoise", (-680, 1200), outputs=[("NOISE", "NOISE")], widgets=[SEED, "randomize"])
    ksamp = G.node("SamplerCustomAdvanced", (-1000, 1340),
                   inputs=[("noise", "NOISE"), ("guider", "GUIDER"), ("sampler", "SAMPLER"),
                           ("sigmas", "SIGMAS"), ("latent_image", "LATENT")],
                   outputs=[("output", "LATENT"), ("denoised_output", "LATENT")])
    G.link(noise, "NOISE", ksamp, "noise", "NOISE")
    G.link(guider, "GUIDER", ksamp, "guider", "GUIDER")
    G.link(samp, "SAMPLER", ksamp, "sampler", "SAMPLER")
    G.link(sig_sw, "output", ksamp, "sigmas", "SIGMAS")
    G.link(lat_sw, "output", ksamp, "latent_image", "LATENT")

    dec = G.node("VAEDecode", (-1000, 1500), inputs=[("samples", "LATENT"), ("vae", "VAE")],
                 outputs=[("IMAGE", "IMAGE")])
    G.link(ksamp, "output", dec, "samples", "LATENT")
    G.link(vae, "VAE", dec, "vae", "VAE")

    crop = crop_to_1080(G, tw, th, dec, "IMAGE", (-680, 1660))
    save = G.node("SaveImage", (-1000, 1660), inputs=[("images", "IMAGE")],
                  widgets=["Ideogram_1080p"], title="save filename", tunable=True)
    G.link(crop, "IMAGE", save, "images", "IMAGE")
    preview = G.node("PreviewImage", (-680, 1800), inputs=[("images", "IMAGE")], title="preview")
    G.link(crop, "IMAGE", preview, "images", "IMAGE")

    # Loaders/guider/schedulers/sampler/decode/crop live inside the box; image seed +
    # toggle are real external nodes (see above). 5 widgets proxied: (pos,neg) /
    # (width,height,save filename).
    proxy_order = [pos, neg, tw, th, save]
    ext_nodes = [ref_img]
    wires = [(ref_img, "IMAGE", ref_lat, "pixels"),
             (ref_img, "ENABLE", lat_sw, "switch"),
             (ref_img, "ENABLE", sig_sw, "switch")]
    sg = emit_subgraph_with_image_seeds(G, "Image Generation", proxy_order, ext_nodes, wires)
    dump_subgraph(sg, "Ideogram-Image-Gen-1080p")


# --------------------------------------------------------------------------------------
# 2. Flux-Inpaint-1080p
# --------------------------------------------------------------------------------------

def build_inpaint():
    G = Graph()
    tw, th = res_primitives(G)
    flux = flux_fill_loaders(G, device=GPU_INPAINT)

    # LoadImage stays OUTSIDE the collapsed box as a real node (see
    # emit_subgraph_with_image_seed docstring) -- proxying it broke the mask editor for
    # already-uploaded files. Built in a throwaway Graph so it's never swept into G's
    # internal node list.
    img = Graph().node("LoadImage", (0, 0),
                       inputs=[("image", "COMBO", "image"), ("upload", "IMAGEUPLOAD", "upload")],
                       outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")], widgets=["example.png", "image"],
                       title="image seed (paint mask here)")

    # scale image + mask to target working resolution (mask via image round-trip)
    # crop="center" = scale-to-cover + center-crop (preserves aspect, no stretch). Applied
    # identically to image and mask so they stay pixel-aligned. img/mask inputs are left
    # unlinked here -- emit_subgraph_with_image_seed wires them from the external node.
    img_s = G.node("ImageScale", (-680, 0),
                   inputs=[("image", "IMAGE"), ("width", "INT", "width"), ("height", "INT", "height")],
                   outputs=[("IMAGE", "IMAGE")], widgets=["lanczos", OUT_W, OUT_H, "center"])
    G.link(tw, "INT", img_s, "width", "INT")
    G.link(th, "INT", img_s, "height", "INT")
    m2i = G.node("MaskToImage", (-680, 160), inputs=[("mask", "MASK")], outputs=[("IMAGE", "IMAGE")])
    m_s = G.node("ImageScale", (-680, 320),
                 inputs=[("image", "IMAGE"), ("width", "INT", "width"), ("height", "INT", "height")],
                 outputs=[("IMAGE", "IMAGE")], widgets=["nearest-exact", OUT_W, OUT_H, "center"])
    G.link(m2i, "IMAGE", m_s, "image", "IMAGE")
    G.link(tw, "INT", m_s, "width", "INT")
    G.link(th, "INT", m_s, "height", "INT")
    i2m = G.node("ImageToMask", (-680, 480), inputs=[("image", "IMAGE")],
                 outputs=[("MASK", "MASK")], widgets=["red"])
    G.link(m_s, "IMAGE", i2m, "image", "IMAGE")
    grow = G.node("GrowMask", (-680, 620), inputs=[("mask", "MASK")], outputs=[("MASK", "MASK")],
                  widgets=[6, 0])
    G.link(i2m, "MASK", grow, "mask", "MASK")

    # denoise=0.65 (not 1.0): at full denoise the masked pixels are pure noise going into
    # the sampler, so the model has no anchor on the ORIGINAL shape under the mask and is
    # free to invent a different one from the prompt alone -- confirmed live (asking for
    # "same turban outlook, just recolor" still changed the turban's shape at denoise=1.0).
    # 0.65 keeps the original structure while still letting color/detail shift with the
    # prompt; exposed as a tunable so it can be raised for bigger edits or lowered further
    # for subtler ones.
    dec, pos, neg, ksamp = flux_fill_body(G, flux, tw, th, img_s, "IMAGE", grow, "MASK",
                                          x=-300, denoise=0.65)
    crop = crop_to_1080(G, tw, th, dec, "IMAGE", (20, 900))
    save = G.node("SaveImage", (-300, 920), inputs=[("images", "IMAGE")],
                  widgets=["Flux_Inpaint_1080p"], title="save filename", tunable=True)
    G.link(crop, "IMAGE", save, "images", "IMAGE")
    preview = G.node("PreviewImage", (20, 1040), inputs=[("images", "IMAGE")], title="preview")
    G.link(crop, "IMAGE", preview, "images", "IMAGE")

    # Loaders/mask pipeline/sampler/decode/crop live inside the box; the image seed is a
    # real external node (see above). 4 proxied groups: (pos,neg) / (denoise) /
    # (width,height,save).
    proxy_order = [pos, neg, ksamp, tw, th, save]
    wires = [("IMAGE", img_s, "image"), ("MASK", m2i, "mask")]
    sg = emit_subgraph_with_image_seed(G, "Flux Inpaint", proxy_order, img, wires)
    dump_subgraph(sg, "Flux-Inpaint-1080p")


# --------------------------------------------------------------------------------------
# 3. Flux-Outpaint-1080p
# --------------------------------------------------------------------------------------

def build_outpaint():
    G = Graph()
    tw, th = res_primitives(G)
    flux = flux_fill_loaders(G, device=GPU_OUTPAINT)

    # LoadImage stays OUTSIDE the collapsed box as a real node (see
    # emit_subgraph_with_image_seed docstring) -- proxying it broke reliable
    # preview/thumbnail behavior for already-uploaded files, same root cause as Inpaint.
    img = Graph().node("LoadImage", (0, 0),
                       inputs=[("image", "COMBO", "image"), ("upload", "IMAGEUPLOAD", "upload")],
                       outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")], widgets=["example.png", "image"],
                       title="image seed (to extend)")

    # No stretch: fit the source inside the target canvas (preserving aspect), THEN pad
    # the remainder. Source can exceed target in either dimension (e.g. a 1023x1537
    # portrait photo into a 1920x1088 landscape target -- confirmed live: assuming the
    # source always fits and just doing (target-source)/2 produced a NEGATIVE pad amount,
    # which ImagePadForOutpaint can't handle (it can only add border, not crop) and crashed
    # with a tensor-shape mismatch). scale = min(1, target_w/src_w, target_h/src_h) is
    # clamped to never exceed 1 -- a source that already fits inside the target gets
    # padded at native resolution unchanged (scale=1, no-op); only a source that's larger
    # in some dimension gets shrunk (uniformly, so it's never stretched) to fit first.
    # min/max are genuine supported functions in ComfyUI's core ComfyMathExpression
    # (comfy_extras/nodes_math.py: MATH_FUNCTIONS = {"min": min, "max": max, "abs": abs,
    # ...}) -- verified against source, not guessed.
    gis = G.node("GetImageSize", (-1000, -320), inputs=[("image", "IMAGE")],
                 outputs=[("width", "INT"), ("height", "INT"), ("batch_size", "INT")])
    scale_expr = G.node("ComfyMathExpression", (-1000, -160), title="outpaint fit-scale (never upscale)")
    sa = scale_expr.add_input("values.a", "FLOAT,INT,BOOLEAN")
    sb = scale_expr.add_input("values.b", "FLOAT,INT,BOOLEAN", shape=7)
    sc = scale_expr.add_input("values.c", "FLOAT,INT,BOOLEAN", shape=7)
    sd = scale_expr.add_input("values.d", "FLOAT,INT,BOOLEAN", shape=7)
    scale_expr.add_input("expression", "STRING", widget="expression")
    scale_expr.add_output("FLOAT", "FLOAT")
    scale_expr.add_output("INT", "INT")
    scale_expr.add_output("BOOL", "BOOLEAN")
    scale_expr.widgets_values = ["min(1, a/b, c/d)"]
    G.link(tw, "INT", scale_expr, sa, "FLOAT,INT,BOOLEAN")
    G.link(gis, "width", scale_expr, sb, "FLOAT,INT,BOOLEAN")
    G.link(th, "INT", scale_expr, sc, "FLOAT,INT,BOOLEAN")
    G.link(gis, "height", scale_expr, sd, "FLOAT,INT,BOOLEAN")

    fit = G.node("ImageScaleBy", (-680, -320),
                 inputs=[("image", "IMAGE"), ("scale_by", "FLOAT", "scale_by")],
                 outputs=[("IMAGE", "IMAGE")], widgets=["lanczos", 1.0])
    G.link(scale_expr, "FLOAT", fit, "scale_by", "FLOAT")
    gis2 = G.node("GetImageSize", (-340, -320), inputs=[("image", "IMAGE")],
                  outputs=[("width", "INT"), ("height", "INT"), ("batch_size", "INT")])
    G.link(fit, "IMAGE", gis2, "image", "IMAGE")

    # max(0, ...) defensively clamps against any 1px rounding mismatch between our scale
    # computation and ImageScaleBy's own internal round() -- never pass a negative pad.
    left = math_expr(G, (-680, 0), "max(0, (a-b)//2)", tw, "INT", "pad left = max(0,(target_w-w)//2)",
                     b_src=gis2, b_slot="width")
    right = math_expr(G, (-680, 140), "max(0, (a-b)-((a-b)//2))", tw, "INT", "pad right", b_src=gis2, b_slot="width")
    top = math_expr(G, (-680, 280), "max(0, (a-b)//2)", th, "INT", "pad top = max(0,(target_h-h)//2)",
                    b_src=gis2, b_slot="height")
    bottom = math_expr(G, (-680, 420), "max(0, (a-b)-((a-b)//2))", th, "INT", "pad bottom", b_src=gis2, b_slot="height")
    pad = G.node("ImagePadForOutpaint", (-340, 0),
                 inputs=[("image", "IMAGE"), ("left", "INT", "left"), ("top", "INT", "top"),
                         ("right", "INT", "right"), ("bottom", "INT", "bottom")],
                 outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")], widgets=[0, 0, 0, 0, 40])
    G.link(fit, "IMAGE", pad, "image", "IMAGE")
    G.link(left, "INT", pad, "left", "INT")
    G.link(top, "INT", pad, "top", "INT")
    G.link(right, "INT", pad, "right", "INT")
    G.link(bottom, "INT", pad, "bottom", "INT")

    # padded canvas is already target resolution -> straight to Flux Fill, then crop to 1080.
    # denoise stays at the default 1.0 (full regen, not exposed) -- unlike inpaint, the
    # padded border has NO original content to preserve, so full denoise is correct here.
    # seed_control="randomize" (unlike inpaint's default "fixed"): a large outpaint can
    # leave most of the canvas newly generated, and a fixed seed reproduces the exact same
    # result (including any hallucination artifact) every run -- randomizing means each
    # run gets a fresh attempt instead of being stuck with one bad seed.
    dec, pos, neg, _ksamp = flux_fill_body(G, flux, tw, th, pad, "IMAGE", pad, "MASK",
                                           x=-300 + 320, seed_control="randomize")
    crop = crop_to_1080(G, tw, th, dec, "IMAGE", (340, 900))
    save = G.node("SaveImage", (20, 920), inputs=[("images", "IMAGE")],
                  widgets=["Flux_Outpaint_1080p"], title="save filename", tunable=True)
    G.link(crop, "IMAGE", save, "images", "IMAGE")
    preview = G.node("PreviewImage", (340, 1040), inputs=[("images", "IMAGE")], title="preview")
    G.link(crop, "IMAGE", preview, "images", "IMAGE")

    # Pad amounts are auto-computed from source size (no manual pad tunables needed).
    # Image seed is a real external node (fans out to both gis.image and fit.image from
    # the SAME exposed input socket -- fit's OWN output then feeds pad internally, so pad
    # no longer needs to be wired to the external node directly). 3 proxied groups:
    # (pos,neg) / (width,height,save).
    proxy_order = [pos, neg, tw, th, save]
    wires = [("IMAGE", gis, "image"), ("IMAGE", fit, "image")]
    sg = emit_subgraph_with_image_seed(G, "Flux Outpaint", proxy_order, img, wires)
    dump_subgraph(sg, "Flux-Outpaint-1080p")


# --------------------------------------------------------------------------------------
# 4. LTX-2.3-Video-Gen-1080p
# --------------------------------------------------------------------------------------

def _guide_stage(G, stage, x, y0, cond, base_lat, frames, ltx):
    """Chain gated LTXVAddGuide lanes (mid/last -- first is handled separately via
    LTXVImgToVideoConditionOnly, see build_video()'s docstring note on GH#12832). Returns
    (final pos/neg/lat switch nodes, {name: guide_node}, {name: (swp, swn, swl)}). Each
    guide's "image" input AND each switch's "switch" (toggle) input are left UNLINKED
    here -- frames[name][0] and the gen toggles are real EXTERNAL nodes (see mask-editor
    gotcha + toggle-visibility gotcha), so the caller wires them in via
    emit_subgraph_with_image_seeds instead of a plain G.link."""
    cur_pos, cur_neg, cur_lat = cond, cond, base_lat
    pos_slot, neg_slot = "positive", "negative"
    if base_lat.type in ("EmptyLTXVLatentVideo", "LTXVLatentUpsampler"):
        lat_slot = "LATENT"
    elif base_lat.type == "LTXVImgToVideoConditionOnly":
        lat_slot = "latent"
    else:
        lat_slot = "output"
    guides = {}
    switches = {}
    for name in ("mid", "last"):
        img, idx, s1, s2, _ = frames[name]
        strength = s1 if stage == "stage1" else s2
        idx_literal = idx if isinstance(idx, int) else 0
        guide = G.node("LTXVAddGuide", (x, y0),
                       inputs=[("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
                               ("vae", "VAE"), ("latent", "LATENT"), ("image", "IMAGE")],
                       outputs=[("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("latent", "LATENT")],
                       widgets=[idx_literal, strength], title=f"{stage} guide {name}")
        fi = guide.add_input("frame_idx", "INT", widget="frame_idx")
        if not isinstance(idx, int):
            G.link(idx, "INT", guide, fi, "INT")
        G.link(cur_pos, pos_slot, guide, "positive", "CONDITIONING")
        G.link(cur_neg, neg_slot, guide, "negative", "CONDITIONING")
        G.link(ltx["vae"], "VAE", guide, "vae", "VAE")
        G.link(cur_lat, lat_slot, guide, "latent", "LATENT")
        guides[name] = guide
        swp = G.node("ComfySwitchNode", (x + 360, y0),
                     inputs=[("switch", "BOOLEAN", "switch"), ("on_true", "CONDITIONING"), ("on_false", "CONDITIONING")],
                     outputs=[("output", "CONDITIONING")], widgets=[False])
        swn = G.node("ComfySwitchNode", (x + 360, y0 + 120),
                     inputs=[("switch", "BOOLEAN", "switch"), ("on_true", "CONDITIONING"), ("on_false", "CONDITIONING")],
                     outputs=[("output", "CONDITIONING")], widgets=[False])
        swl = G.node("ComfySwitchNode", (x + 360, y0 + 240),
                     inputs=[("switch", "BOOLEAN", "switch"), ("on_true", "LATENT"), ("on_false", "LATENT")],
                     outputs=[("output", "LATENT")], widgets=[False])
        switches[name] = (swp, swn, swl)
        G.link(guide, "positive", swp, "on_true", "CONDITIONING")
        G.link(cur_pos, pos_slot, swp, "on_false", "CONDITIONING")
        G.link(guide, "negative", swn, "on_true", "CONDITIONING")
        G.link(cur_neg, neg_slot, swn, "on_false", "CONDITIONING")
        G.link(guide, "latent", swl, "on_true", "LATENT")
        G.link(cur_lat, lat_slot, swl, "on_false", "LATENT")
        cur_pos, cur_neg, cur_lat = swp, swn, swl
        pos_slot = neg_slot = lat_slot = "output"
        y0 += 380
    return cur_pos, cur_neg, cur_lat, guides, switches


def build_video():
    G = Graph()
    # resolution + fps + duration primitives
    tw = G.node("PrimitiveInt", (-2200, 0), outputs=[("INT", "INT")], widgets=[OUT_W],
                title="target width", tunable=True)
    th = G.node("PrimitiveInt", (-2200, 120), outputs=[("INT", "INT")], widgets=[OUT_H],
                title="target height", tunable=True)
    # stage1 (LTX's internal half-res pass) is computed from target width/height, not a
    # separate manual control -- user asked for exactly ONE width/height pair exposed.
    s1w = math_expr(G, (-2200, 240), "a//2", tw, "INT", title="stage1 width = target/2")
    s1h = math_expr(G, (-2200, 360), "a//2", th, "INT", title="stage1 height = target/2")
    fps = G.node("PrimitiveFloat", (-2200, 480), outputs=[("FLOAT", "FLOAT")],
                 widgets=[float(CONFIG["fps"])], title="fps", tunable=True)
    seconds = round(CONFIG["num_frames"] / CONFIG["fps"], 3)
    dur = G.node("PrimitiveFloat", (-2200, 600), outputs=[("FLOAT", "FLOAT")],
                 widgets=[seconds], title="duration (seconds)", tunable=True)
    nframes = math_expr(G, (-2200, 720), "(a*b-1)//8*8+1", dur, "FLOAT",
                        title="duration -> num_frames (8k+1 safe)", b_src=fps, b_slot="FLOAT")
    res = {"out_w": tw, "out_h": th, "s1_w": s1w, "s1_h": s1h, "fps": fps, "num_frames": nframes}

    c = C_LTX
    x = 900
    # Pinned to a dedicated GPU (see GPU_VIDEO). NOTE: ComfyUI-MultiGPU (as installed) only
    # wraps CheckpointLoaderSimple among this stage's loaders -- confirmed via its actual
    # source, no *MultiGPU variant exists for LTXVAudioVAELoader, LatentUpscaleModelLoader,
    # or LTXAVTextEncoderLoader (LTX-2.3-specific nodes newer than that pack's LTX
    # integration). Those three stay on whatever ComfyUI's own default device is; only the
    # main ~22B-parameter LTX checkpoint (the largest, most impactful piece) gets isolated.
    ckpt = G.node("CheckpointLoaderSimpleMultiGPU", (x, 0),
                  outputs=[("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE")],
                  widgets=[c["checkpoint"], GPU_VIDEO])
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
                 widgets=["describe the motion/action + audio for the video"], title="positive prompt",
                 tunable=True)
    G.link(enc, "CLIP", pos, "clip", "CLIP")
    neg = G.node("CLIPTextEncode", (x, 1120), inputs=[("clip", "CLIP")],
                 outputs=[("CONDITIONING", "CONDITIONING")],
                 widgets=["pc game, console game, video game, cartoon, childish, ugly"], title="negative prompt",
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
                     outputs=[("LATENT", "LATENT")], widgets=[S1_W, S1_H, CONFIG["num_frames"], 1])
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

    mid_idx = math_expr(G, (x - 320, 1980), "((a-1)//2)//8*8", res["num_frames"], "INT",
                        title="mid frame index (live, 8k-aligned)")

    # Optional keyframe lanes -- ONE node each. LoadImageWithEnable (custom node,
    # generator/custom_nodes/ltx23_helpers) bundles the image picker + an enable checkbox,
    # inheriting LoadImage's upload button / preview thumbnail (and mask-editor, though
    # video keyframes don't need it). Its ENABLE output fans to the same Switch consumers
    # the separate toggles used to. No stock/installed node combines an image picker with
    # a boolean toggle (verified across all 1202 live /object_info types), so this small
    # custom node is the only literal way to get "image + toggle in one node" -- i.e. the
    # 3 external input nodes the user asked for (was 6: 3 LoadImage + 3 PrimitiveBoolean).
    # First/last default OFF, mid follows CONFIG (unchanged from before). Built in a SHARED
    # throwaway Graph() so all three get unique sequential ids (1-3).
    EG = Graph()
    # Widget ORDER for LoadImageWithEnable is [image, enable, upload] (verified live against
    # the frontend's own canonical serialization -- the upload widget is appended AFTER the
    # declared widgets). So widgets_values must be [filename, enable_bool, "image"]; putting
    # the upload string at index 1 would land on `enable` and coerce to truthy True, silently
    # forcing every lane ON (the bug that made mid/last keyframes apply despite "off").
    _li_inputs = [("image", "COMBO", "image"), ("enable", "BOOLEAN", "enable"),
                  ("upload", "IMAGEUPLOAD", "upload")]
    _li_outputs = [("IMAGE", "IMAGE"), ("MASK", "MASK"), ("ENABLE", "BOOLEAN")]
    first_img = EG.node("LoadImageWithEnable", (0, 0), inputs=_li_inputs, outputs=_li_outputs,
                        widgets=["example.png", False, "image"], title="first frame")
    mid_img = EG.node("LoadImageWithEnable", (340, 0), inputs=_li_inputs, outputs=_li_outputs,
                      widgets=["example.png", CONFIG["gen_mid"], "image"], title="mid frame")
    last_img = EG.node("LoadImageWithEnable", (680, 0), inputs=_li_inputs, outputs=_li_outputs,
                       widgets=["example.png", False, "image"], title="last frame")

    # First-frame conditioning uses LTXVImgToVideoConditionOnly (noise-mask based) instead
    # of LTXVAddGuide, matching Lightricks' own official two-stage reference template
    # EXACTLY (LTX-2.3_T2V_I2V_Two_Stage_Distilled.json) -- confirmed live via /object_info
    # and by reading that template's actual node graph. This sidesteps a known, still-open
    # bug: Comfy-Org/ComfyUI#12832 "LTXVAddGuide node last 1 second very bad", an EXACT
    # match for our setup (first=0, last=-1 frame_idx) and the tail-end camera-cut/color
    # artifact confirmed live in this project. LTXVImgToVideoConditionOnly is documented as
    # first-frames-only, so mid/last still use LTXVAddGuide (no official alternative
    # exists for those positions -- accept the known-bug risk there).
    #
    # Official strengths (0.7 stage1, 1.0 stage2) are used as-is -- ImgToVideoConditionOnly
    # caps strength at 1.0 (unlike LTXVAddGuide's 0-10 range), so our earlier "raised
    # strength" values don't carry over; these are the proven, tested defaults instead.
    STRENGTH_FIRST_S1, STRENGTH_FIRST_S2 = 0.7, 1.0
    # Last-frame guide strength, [stage1, stage2]. Was (1.4, 1.8) -- over-raised during the
    # earlier "keyframes not respected" fix; at that level LTX's temporal attention bleeds the
    # last-keyframe pull across the whole back half of the latent and the model freezes on it
    # (confirmed live: the back ~2s of a 4s clip decoded as a frozen last-keyframe still).
    # Back to the node's default 1.0 for both stages: respected but not frozen. stage2's value
    # is proxied on the box (s2_guides["last"] in proxy_order), so it's tunable per-run --
    # raise it if the last keyframe isn't sticking, lower it if the tail freezes.
    STRENGTH_LAST = (1.0, 1.0)
    # Toggles for first/last default to OFF (was True on first/last originally) -- nothing
    # happens unless explicitly turned on, which avoids the still-unset example.png
    # placeholder getting used as a real keyframe just because a fresh workflow's defaults
    # happened to be ON. Matches the official template's own bypass_i2v default (True =
    # bypassed) too.
    frames = {
        "mid": (mid_img, mid_idx, CONFIG["strength_mid"][0], CONFIG["strength_mid"][1], CONFIG["gen_mid"]),
        "last": (last_img, -1, STRENGTH_LAST[0], STRENGTH_LAST[1], False),
    }

    # First-frame preprocessing chain, matching the official template: resize (longer edge
    # 1536, lanczos) once, feeding BOTH stage1 (via LTXVPreprocess, compression=18) and
    # stage2 (direct, no preprocess) conditioning. `not_first` inverts the first-frame
    # node's ENABLE checkbox into the `bypass` boolean ImgToVideoConditionOnly expects
    # (bypass=True means "don't use it", the opposite of ENABLE's semantics).
    resize_first = G.node("ResizeImageMaskNode", (x - 700, 200),
                          inputs=[("input", "IMAGE,MASK")], outputs=[("resized", "IMAGE")],
                          widgets=["scale longer dimension", 1536, "lanczos"])
    preprocess_first = G.node("LTXVPreprocess", (x - 700, 340),
                              inputs=[("image", "IMAGE"), ("img_compression", "INT")],
                              outputs=[("output_image", "IMAGE")], widgets=[18])
    G.link(resize_first, "resized", preprocess_first, "image", "IMAGE")
    not_first = G.node("ComfyNotNode", (x - 700, 460), inputs=[("value", "*")],
                       outputs=[("BOOLEAN", "BOOLEAN")])

    cond_first_s1 = G.node("LTXVImgToVideoConditionOnly", (x - 320, 1480),
                           inputs=[("vae", "VAE"), ("image", "IMAGE"), ("latent", "LATENT"),
                                   ("strength", "FLOAT", "strength"), ("bypass", "BOOLEAN", "bypass")],
                           outputs=[("latent", "LATENT")], widgets=[STRENGTH_FIRST_S1, False],
                           title="stage1 first-frame condition")
    G.link(ckpt, "VAE", cond_first_s1, "vae", "VAE")
    G.link(preprocess_first, "output_image", cond_first_s1, "image", "IMAGE")
    G.link(vid_lat, "LATENT", cond_first_s1, "latent", "LATENT")
    G.link(not_first, "BOOLEAN", cond_first_s1, "bypass", "BOOLEAN")

    ltx = {"vae": ckpt, "lora": lora, "audio_vae": audio_vae, "upscaler": upscaler}
    s1_pos, s1_neg, s1_lat, s1_guides, s1_switches = _guide_stage(G, "stage1", x + 500, 0, cond, cond_first_s1, frames, ltx)

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
    noise1 = G.node("RandomNoise", (x + 1100, 420), outputs=[("NOISE", "NOISE")], widgets=[SEED, "randomize"])
    sig1 = G.node("ManualSigmas", (x + 1100, 540), outputs=[("SIGMAS", "SIGMAS")], widgets=[SIGMAS_S1])
    ks1 = G.node("SamplerCustomAdvanced", (x + 1100, 680),
                 inputs=[("noise", "NOISE"), ("guider", "GUIDER"), ("sampler", "SAMPLER"),
                         ("sigmas", "SIGMAS"), ("latent_image", "LATENT")],
                 outputs=[("output", "LATENT"), ("denoised_output", "LATENT")])
    dst_types = {"noise": "NOISE", "guider": "GUIDER", "sampler": "SAMPLER", "sigmas": "SIGMAS", "latent_image": "LATENT"}
    for src, slot, dst in [(noise1, "NOISE", "noise"), (g1, "GUIDER", "guider"), (samp1, "SAMPLER", "sampler"),
                           (sig1, "SIGMAS", "sigmas"), (concat1, "latent", "latent_image")]:
        G.link(src, slot, ks1, dst, dst_types[dst])
    sep1 = G.node("LTXVSeparateAVLatent", (x + 1100, 860), inputs=[("av_latent", "LATENT")],
                  outputs=[("video_latent", "LATENT"), ("audio_latent", "LATENT")])
    G.link(ks1, "output", sep1, "av_latent", "LATENT")

    # Strips stale keyframe-guide metadata (and any temporal padding LTXVAddGuide injected
    # to encode the reference image) from stage1's output BEFORE it's upsampled -- confirmed
    # via a live diff against the user's own known-good reference workflow
    # (LTS-2.3-1080p.json) that it does this exact crop at this exact point (after
    # LTXVSeparateAVLatent, before the latent is reused) and does NOT exhibit the tail-end
    # camera-cut/color-flicker artifact we hit without it. We don't otherwise reuse stage1's
    # conditioning (stage2 rebuilds its own guides from `cond`), so only the cropped LATENT
    # output matters here -- the conditioning outputs are unused, which is fine.
    crop_guides = G.node("LTXVCropGuides", (x + 1300, 860),
                         inputs=[("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
                                 ("latent", "LATENT")],
                         outputs=[("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
                                  ("latent", "LATENT")])
    G.link(s1_pos, "output", crop_guides, "positive", "CONDITIONING")
    G.link(s1_neg, "output", crop_guides, "negative", "CONDITIONING")
    G.link(sep1, "video_latent", crop_guides, "latent", "LATENT")

    up = G.node("LTXVLatentUpsampler", (x + 1500, 0),
                inputs=[("samples", "LATENT"), ("upscale_model", "LATENT_UPSCALE_MODEL"), ("vae", "VAE")],
                outputs=[("LATENT", "LATENT")])
    G.link(crop_guides, "latent", up, "samples", "LATENT")
    G.link(upscaler, "LATENT_UPSCALE_MODEL", up, "upscale_model", "LATENT_UPSCALE_MODEL")
    G.link(ckpt, "VAE", up, "vae", "VAE")

    # Stage2's first-frame conditioning reuses resize_first's DIRECT output (no
    # LTXVPreprocess -- matches the official template, which only preprocesses for
    # stage1) at full strength (1.0) on the freshly-upsampled latent. tunable=True so its
    # "strength" widget can be proxied below.
    cond_first_s2 = G.node("LTXVImgToVideoConditionOnly", (x + 1700, 200),
                           inputs=[("vae", "VAE"), ("image", "IMAGE"), ("latent", "LATENT"),
                                   ("strength", "FLOAT", "strength"), ("bypass", "BOOLEAN", "bypass")],
                           outputs=[("latent", "LATENT")], widgets=[STRENGTH_FIRST_S2, False],
                           title="stage2 first-frame condition", tunable=True)
    G.link(ckpt, "VAE", cond_first_s2, "vae", "VAE")
    G.link(resize_first, "resized", cond_first_s2, "image", "IMAGE")
    G.link(up, "LATENT", cond_first_s2, "latent", "LATENT")
    G.link(not_first, "BOOLEAN", cond_first_s2, "bypass", "BOOLEAN")

    s2_pos, s2_neg, s2_lat, s2_guides, s2_switches = _guide_stage(G, "stage2", x + 1900, 0, cond, cond_first_s2, frames, ltx)
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
    # Both noise seeds randomize (was "fixed") -- with a fixed seed + identical inputs,
    # ComfyUI caches and returns the EXACT same video every queue (user hit this: 3 queued
    # jobs -> 3 identical outputs). Randomize gives a different video per run. To reproduce
    # a good one, switch back to "fixed" and paste the seed.
    noise2 = G.node("RandomNoise", (x + 2400, 420), outputs=[("NOISE", "NOISE")], widgets=[SEED + 1, "randomize"])
    sig2 = G.node("ManualSigmas", (x + 2400, 540), outputs=[("SIGMAS", "SIGMAS")], widgets=[SIGMAS_S2])
    ks2 = G.node("SamplerCustomAdvanced", (x + 2400, 680),
                 inputs=[("noise", "NOISE"), ("guider", "GUIDER"), ("sampler", "SAMPLER"),
                         ("sigmas", "SIGMAS"), ("latent_image", "LATENT")],
                 outputs=[("output", "LATENT"), ("denoised_output", "LATENT")])
    for src, slot, dst in [(noise2, "NOISE", "noise"), (g2, "GUIDER", "guider"), (samp2, "SAMPLER", "sampler"),
                           (sig2, "SIGMAS", "sigmas"), (concat2, "latent", "latent_image")]:
        G.link(src, slot, ks2, dst, dst_types[dst])
    sep2 = G.node("LTXVSeparateAVLatent", (x + 2400, 860), inputs=[("av_latent", "LATENT")],
                  outputs=[("video_latent", "LATENT"), ("audio_latent", "LATENT")])
    G.link(ks2, "output", sep2, "av_latent", "LATENT")

    # NOTE: stage2's guide padding is NOT cropped here before decode. A stage2 LTXVCropGuides
    # was tried and REVERTED -- LTXVCropGuides crops only from the END of the latent
    # (latent[:,:,:-num_keyframes]), which is correct for a last-frame guide but WRONG for a
    # mid-frame guide (its padding sits in the middle, so end-cropping chops real frames).
    # With mid enabled that produced a 161-frame broken output. The cost of NOT cropping
    # stage2 is a minor ~8-frame end-hold on the last keyframe (0.33s) when the last frame is
    # used -- acceptable, and far safer than the mid-interaction failure mode. stage1's
    # crop_guides (above) stays, since stage1's cropped latent is reused by the upscaler.
    vdec = G.node("LTXVTiledVAEDecode", (x + 2900, 0), inputs=[("vae", "VAE"), ("latents", "LATENT")],
                  outputs=[("image", "IMAGE")], widgets=[2, 2, 6, False, "auto", "auto"])
    G.link(ckpt, "VAE", vdec, "vae", "VAE")
    G.link(sep2, "video_latent", vdec, "latents", "LATENT")
    adec = G.node("LTXVAudioVAEDecode", (x + 2900, 200), inputs=[("samples", "LATENT"), ("audio_vae", "VAE")],
                  outputs=[("Audio", "AUDIO")])
    G.link(sep2, "audio_latent", adec, "samples", "LATENT")
    G.link(audio_vae, "Audio VAE", adec, "audio_vae", "VAE")

    # 4px equal crop top/bottom on the decoded frames -> absolute 1080p, then mux
    crop = crop_to_1080(G, res["out_w"], res["out_h"], vdec, "image", (x + 2900, 400))
    cvid = G.node("CreateVideo", (x + 3300, 400),
                  inputs=[("images", "IMAGE"), ("audio", "AUDIO"), ("fps", "FLOAT")],
                  outputs=[("VIDEO", "VIDEO")], widgets=[])
    G.link(crop, "IMAGE", cvid, "images", "IMAGE")
    G.link(adec, "Audio", cvid, "audio", "AUDIO")
    G.link(res["fps"], "FLOAT", cvid, "fps", "FLOAT")
    save = G.node("SaveVideo", (x + 3300, 600), inputs=[("video", "VIDEO")],
                  widgets=["LTX23_VideoGen_1080p", "auto", "auto"])
    G.link(cvid, "VIDEO", save, "video", "VIDEO")

    # Loaders/schedulers/samplers/guides/decode/crop/mux all live inside the box. Keyframe
    # images AND their ON/OFF toggles are real external nodes in the main canvas (see
    # above); only prompts, keyframe strength (first via ImgToVideoConditionOnly's own
    # stage2 strength; last via stage2's guide -- the final/most perceptually impactful
    # pass in both cases; stage1 stays fixed), duration, and width/height/fps are proxied
    # onto the box.
    proxy_order = [pos, neg, cond_first_s2, s2_guides["last"], dur, tw, th, fps]
    ext_nodes = [first_img, mid_img, last_img]
    # Each lane's single node outputs IMAGE (fans to its guide consumers) AND ENABLE
    # (fans to the same switch consumers the separate toggle used to, plus the first-frame
    # inverter). Keyed by (node.id, output_name), so one node's IMAGE + ENABLE get distinct
    # subgraph input sockets while repeated ENABLE wires from the same node collapse onto
    # one shared socket and fan to all its targets.
    wires = [(first_img, "IMAGE", resize_first, "input"),
             (first_img, "ENABLE", not_first, "value")]
    for name, img in (("mid", mid_img), ("last", last_img)):
        wires.append((img, "IMAGE", s1_guides[name], "image"))
        wires.append((img, "IMAGE", s2_guides[name], "image"))
        for sw in s1_switches[name]:
            wires.append((img, "ENABLE", sw, "switch"))
        for sw in s2_switches[name]:
            wires.append((img, "ENABLE", sw, "switch"))
    sg = emit_subgraph_with_image_seeds(G, "Video Generation", proxy_order, ext_nodes, wires)
    dump_subgraph(sg, "LTX-2.3-Video-Gen-1080p")


def build_sdxl_flux_outpaint():
    """5-stage seamless outpaint at 1080p: SDXL+ControlNet Union (repaint) -> restore
    original center -> Flux Fill repaint of the border -> crop to 1920x1080.

    Implements the "Seamless Outpainting with Flux in ComfyUI" flow: direct Flux outpaint
    is weak, so SDXL + ControlNet Union promax (set to the "repaint" type) handles the
    initial outpaint; the original center is composited back to restore exact details and
    kill the seam; then a light Flux Fill pass (denoise 0.35) repaints just the border to
    fix SDXL artifacts. The SDXL/ControlNet wiring follows xinsir's own
    ControlNetPlus/workflow/outpaint.json reference exactly.

    Canvas runs at full 1920x1088 (cropped to 1080p) -- NOT a smaller SDXL-friendly size +
    upscale, because shrinking a high-res source to ~576px then upscaling produced weak,
    flat outpainted borders (a visible "frame" around the original). Full res keeps the
    source at native detail. (Caveat: SDXL is trained near 1MP, so 2MP is above its sweet
    spot; for outpaint -- extending backgrounds, not full scenes -- this is acceptable, and
    the Flux repaint pass cleans up artifacts.)

    REQUIRES 2 downloaded models (both exposed as tunable pickers): an SDXL checkpoint
    (C_SDXL.checkpoint) and ControlNet Union SDXL promax (C_SDXL.controlnet). Flux Fill /
    clip_l / t5xxl / ae are already on the server. Uses VANILLA Flux loaders (no MultiGPU
    device pinning -- see the flux_fill_loaders device-widget note).
    """
    G = Graph()
    # Canvas runs at full output resolution (OUT_W x OUT_H = 1920x1088, cropped to 1080p at
    # the end). Both tunable on the box. (Was 1024x576 + upscale -- that shrank a high-res
    # source to 576px and produced weak, flat outpainted borders. Full-res fixes that.)
    tw = G.node("PrimitiveInt", (-2200, 0), outputs=[("INT", "INT")], widgets=[OUT_W],
                title="canvas width", tunable=True)
    th = G.node("PrimitiveInt", (-2200, 120), outputs=[("INT", "INT")], widgets=[OUT_H],
                title="canvas height", tunable=True)
    # ONE prompt pair drives BOTH the SDXL and Flux text encoders (linked to each
    # CLIPTextEncode.text so the user sets the outpaint prompt once).
    pos_str = G.node("PrimitiveString", (-2200, 280), outputs=[("STRING", "STRING")],
                     widgets=[""], title="positive prompt", tunable=True)
    neg_str = G.node("PrimitiveString", (-2200, 400), outputs=[("STRING", "STRING")],
                     widgets=[""], title="negative prompt", tunable=True)

    img = Graph().node("LoadImage", (0, 0),
                       inputs=[("image", "COMBO", "image"), ("upload", "IMAGEUPLOAD", "upload")],
                       outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")], widgets=["example.png", "image"],
                       title="image to outpaint")

    # ---- Stage 1: fit source inside the canvas (preserve aspect, never upscale), then pad
    # the remainder to the working resolution -> padded IMAGE (source centered) + border MASK.
    # (Same proven block as build_outpaint's stage 1.)
    gis = G.node("GetImageSize", (-1900, -320), inputs=[("image", "IMAGE")],
                 outputs=[("width", "INT"), ("height", "INT"), ("batch_size", "INT")])
    scale_expr = G.node("ComfyMathExpression", (-1900, -160), title="outpaint fit-scale")
    sa = scale_expr.add_input("values.a", "FLOAT,INT,BOOLEAN")
    sb = scale_expr.add_input("values.b", "FLOAT,INT,BOOLEAN", shape=7)
    sc = scale_expr.add_input("values.c", "FLOAT,INT,BOOLEAN", shape=7)
    sd = scale_expr.add_input("values.d", "FLOAT,INT,BOOLEAN", shape=7)
    scale_expr.add_input("expression", "STRING", widget="expression")
    scale_expr.add_output("FLOAT", "FLOAT")
    scale_expr.add_output("INT", "INT")
    scale_expr.add_output("BOOL", "BOOLEAN")
    scale_expr.widgets_values = ["min(1, a/b, c/d)"]
    G.link(tw, "INT", scale_expr, sa, "FLOAT,INT,BOOLEAN")
    G.link(gis, "width", scale_expr, sb, "FLOAT,INT,BOOLEAN")
    G.link(th, "INT", scale_expr, sc, "FLOAT,INT,BOOLEAN")
    G.link(gis, "height", scale_expr, sd, "FLOAT,INT,BOOLEAN")
    fit = G.node("ImageScaleBy", (-1580, -320),
                 inputs=[("image", "IMAGE"), ("scale_by", "FLOAT", "scale_by")],
                 outputs=[("IMAGE", "IMAGE")], widgets=["lanczos", 1.0])
    G.link(scale_expr, "FLOAT", fit, "scale_by", "FLOAT")
    gis2 = G.node("GetImageSize", (-1260, -320), inputs=[("image", "IMAGE")],
                  outputs=[("width", "INT"), ("height", "INT"), ("batch_size", "INT")])
    G.link(fit, "IMAGE", gis2, "image", "IMAGE")
    left = math_expr(G, (-1580, 0), "max(0, (a-b)//2)", tw, "INT", "pad left",
                     b_src=gis2, b_slot="width")
    right = math_expr(G, (-1580, 140), "max(0, (a-b)-((a-b)//2))", tw, "INT", "pad right",
                      b_src=gis2, b_slot="width")
    top = math_expr(G, (-1580, 280), "max(0, (a-b)//2)", th, "INT", "pad top",
                    b_src=gis2, b_slot="height")
    bottom = math_expr(G, (-1580, 420), "max(0, (a-b)-((a-b)//2))", th, "INT", "pad bottom",
                       b_src=gis2, b_slot="height")
    pad = G.node("ImagePadForOutpaint", (-1260, 0),
                 inputs=[("image", "IMAGE"), ("left", "INT", "left"), ("top", "INT", "top"),
                         ("right", "INT", "right"), ("bottom", "INT", "bottom")],
                 outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")], widgets=[0, 0, 0, 0, 80])
    G.link(fit, "IMAGE", pad, "image", "IMAGE")
    G.link(left, "INT", pad, "left", "INT")
    G.link(top, "INT", pad, "top", "INT")
    G.link(right, "INT", pad, "right", "INT")
    G.link(bottom, "INT", pad, "bottom", "INT")
    # padded image = pad:IMAGE ; border mask = pad:MASK

    # ---- Stage 2: SDXL outpaint with ControlNet Union (repaint type) -- xinsir reference.
    ckpt = G.node("CheckpointLoaderSimple", (-1900, 600),
                  outputs=[("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE")],
                  widgets=[C_SDXL["checkpoint"]], title="SDXL checkpoint", tunable=True)
    cn_load = G.node("ControlNetLoader", (-1900, 780), outputs=[("CONTROL_NET", "CONTROL_NET")],
                     widgets=[C_SDXL["controlnet"]], title="ControlNet Union promax", tunable=True)
    cn_type = G.node("SetUnionControlNetType", (-1900, 960),
                     inputs=[("control_net", "CONTROL_NET")], outputs=[("CONTROL_NET", "CONTROL_NET")],
                     widgets=["repaint"], title="set controlnet type = repaint")
    G.link(cn_load, "CONTROL_NET", cn_type, "control_net", "CONTROL_NET")
    # Control image = padded image with the border cleaned to pure black (xinsir's prep):
    # paste MaskToImage(InvertMask(border)) into the border, leaving the original center.
    inv = G.node("InvertMask", (-1580, 600), inputs=[("mask", "MASK")], outputs=[("MASK", "MASK")])
    G.link(pad, "MASK", inv, "mask", "MASK")
    m2i = G.node("MaskToImage", (-1580, 720), inputs=[("mask", "MASK")], outputs=[("IMAGE", "IMAGE")])
    G.link(inv, "MASK", m2i, "mask", "MASK")
    ctrl = G.node("ImageCompositeMasked", (-1260, 600),
                  inputs=[("destination", "IMAGE"), ("source", "IMAGE"), ("mask", "MASK")],
                  outputs=[("IMAGE", "IMAGE")], widgets=[0, 0, False], title="control image")
    G.link(pad, "IMAGE", ctrl, "destination", "IMAGE")
    G.link(m2i, "IMAGE", ctrl, "source", "IMAGE")
    G.link(pad, "MASK", ctrl, "mask", "MASK")
    # SDXL prompts (shared PrimitiveString -> SDXL CLIP).
    sdxl_pos = G.node("CLIPTextEncode", (-1580, 900),
                      inputs=[("clip", "CLIP"), ("text", "STRING", "text")],
                      outputs=[("CONDITIONING", "CONDITIONING")], widgets=[""], title="SDXL positive")
    G.link(ckpt, "CLIP", sdxl_pos, "clip", "CLIP")
    G.link(pos_str, "STRING", sdxl_pos, "text", "STRING")
    sdxl_neg = G.node("CLIPTextEncode", (-1580, 1020),
                      inputs=[("clip", "CLIP"), ("text", "STRING", "text")],
                      outputs=[("CONDITIONING", "CONDITIONING")], widgets=[""], title="SDXL negative")
    G.link(ckpt, "CLIP", sdxl_neg, "clip", "CLIP")
    G.link(neg_str, "STRING", sdxl_neg, "text", "STRING")
    # Apply the typed controlnet (vae input is optional -> omitted, matching the reference).
    cn_apply = G.node("ControlNetApplyAdvanced", (-1260, 900),
                      inputs=[("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
                              ("control_net", "CONTROL_NET"), ("image", "IMAGE")],
                      outputs=[("positive", "CONDITIONING"), ("negative", "CONDITIONING")],
                      widgets=[1.0, 0.0, 1.0])
    G.link(sdxl_pos, "CONDITIONING", cn_apply, "positive", "CONDITIONING")
    G.link(sdxl_neg, "CONDITIONING", cn_apply, "negative", "CONDITIONING")
    G.link(cn_type, "CONTROL_NET", cn_apply, "control_net", "CONTROL_NET")
    G.link(ctrl, "IMAGE", cn_apply, "image", "IMAGE")
    # Encode for inpaint; grow_mask_by widens the regenerated seam-blend zone.
    vae_enc = G.node("VAEEncodeForInpaint", (-1260, 1080),
                     inputs=[("pixels", "IMAGE"), ("vae", "VAE"), ("mask", "MASK")],
                     outputs=[("LATENT", "LATENT")], widgets=[C_SDXL["grow_mask"]])
    G.link(ctrl, "IMAGE", vae_enc, "pixels", "IMAGE")
    G.link(ckpt, "VAE", vae_enc, "vae", "VAE")
    G.link(pad, "MASK", vae_enc, "mask", "MASK")
    ks_sdxl = G.node("KSampler", (-940, 900),
                     inputs=[("model", "MODEL"), ("positive", "CONDITIONING"),
                             ("negative", "CONDITIONING"), ("latent_image", "LATENT")],
                     outputs=[("LATENT", "LATENT")],
                     widgets=[SEED, "randomize", C_SDXL["steps"], C_SDXL["cfg"], C_SDXL["sampler"],
                              C_SDXL["scheduler"], 1.0], title="SDXL outpaint")
    G.link(ckpt, "MODEL", ks_sdxl, "model", "MODEL")
    G.link(cn_apply, "positive", ks_sdxl, "positive", "CONDITIONING")
    G.link(cn_apply, "negative", ks_sdxl, "negative", "CONDITIONING")
    G.link(vae_enc, "LATENT", ks_sdxl, "latent_image", "LATENT")
    sdxl_dec = G.node("VAEDecode", (-940, 1080), inputs=[("samples", "LATENT"), ("vae", "VAE")],
                      outputs=[("IMAGE", "IMAGE")], title="SDXL outpainted")
    G.link(ks_sdxl, "LATENT", sdxl_dec, "samples", "LATENT")
    G.link(ckpt, "VAE", sdxl_dec, "vae", "VAE")

    # ---- Stage 3: paste the original center back -> restore exact details, kill the seam.
    # The pad feathering softens the boundary; mask = inverted border mask (center only).
    restore = G.node("ImageCompositeMasked", (-620, 900),
                     inputs=[("destination", "IMAGE"), ("source", "IMAGE"), ("mask", "MASK")],
                     outputs=[("IMAGE", "IMAGE")], widgets=[0, 0, False], title="restore original center")
    G.link(sdxl_dec, "IMAGE", restore, "destination", "IMAGE")
    G.link(ctrl, "IMAGE", restore, "source", "IMAGE")
    G.link(inv, "MASK", restore, "mask", "MASK")

    # ---- Stage 4: Flux Fill repaint of the border only (fix SDXL artifacts), denoise 0.35.
    flux = flux_fill_loaders(G, x=-620, device=None)
    grow = G.node("GrowMaskWithBlur", (-620, 1080), inputs=[("mask", "MASK")],
                  outputs=[("MASK", "MASK")],
                  widgets=[10, 0.0, True, False, 10.0, 1.0, 1.0, False],
                  title="repaint mask (grow + blur)")
    G.link(pad, "MASK", grow, "mask", "MASK")
    flux_pos = G.node("CLIPTextEncode", (-300, 900),
                      inputs=[("clip", "CLIP"), ("text", "STRING", "text")],
                      outputs=[("CONDITIONING", "CONDITIONING")], widgets=[""], title="Flux positive")
    G.link(flux["clip"], "CLIP", flux_pos, "clip", "CLIP")
    G.link(pos_str, "STRING", flux_pos, "text", "STRING")
    fg = G.node("FluxGuidance", (-300, 1020), inputs=[("conditioning", "CONDITIONING")],
                outputs=[("CONDITIONING", "CONDITIONING")], widgets=[C_FF["guidance"]])
    G.link(flux_pos, "CONDITIONING", fg, "conditioning", "CONDITIONING")
    flux_neg = G.node("CLIPTextEncode", (-300, 1140),
                      inputs=[("clip", "CLIP"), ("text", "STRING", "text")],
                      outputs=[("CONDITIONING", "CONDITIONING")], widgets=[""], title="Flux negative")
    G.link(flux["clip"], "CLIP", flux_neg, "clip", "CLIP")
    G.link(neg_str, "STRING", flux_neg, "text", "STRING")
    inp4 = G.node("InpaintModelConditioning", (-300, 1280),
                  inputs=[("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
                          ("vae", "VAE"), ("pixels", "IMAGE"), ("mask", "MASK")],
                  outputs=[("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
                           ("latent", "LATENT")], widgets=[False])
    G.link(fg, "CONDITIONING", inp4, "positive", "CONDITIONING")
    G.link(flux_neg, "CONDITIONING", inp4, "negative", "CONDITIONING")
    G.link(flux["vae"], "VAE", inp4, "vae", "VAE")
    G.link(restore, "IMAGE", inp4, "pixels", "IMAGE")
    G.link(grow, "MASK", inp4, "mask", "MASK")
    ks4 = G.node("KSampler", (20, 1280),
                 inputs=[("model", "MODEL"), ("positive", "CONDITIONING"),
                         ("negative", "CONDITIONING"), ("latent_image", "LATENT")],
                 outputs=[("LATENT", "LATENT")],
                 widgets=[SEED, "randomize", C_FF["steps"], C_FF["cfg"], "euler", "normal", 0.35],
                 title="Flux repaint (denoise 0.35)", tunable=True)
    G.link(flux["unet"], "MODEL", ks4, "model", "MODEL")
    G.link(inp4, "positive", ks4, "positive", "CONDITIONING")
    G.link(inp4, "negative", ks4, "negative", "CONDITIONING")
    G.link(inp4, "latent", ks4, "latent_image", "LATENT")
    flux_dec = G.node("VAEDecode", (20, 1440), inputs=[("samples", "LATENT"), ("vae", "VAE")],
                      outputs=[("IMAGE", "IMAGE")], title="Flux repainted")
    G.link(ks4, "LATENT", flux_dec, "samples", "LATENT")
    G.link(flux["vae"], "VAE", flux_dec, "vae", "VAE")

    # ---- Stage 5: crop to 1080p (canvas is 1920x1088; shave 4px top/bottom -> 1920x1080)
    # and save. No 4x upscale -- canvas is already full output res (upscale was for the old
    # sub-1080p working size; at full res it's redundant).
    final = crop_to_1080(G, tw, th, flux_dec, "IMAGE", (340, 1500))
    save = G.node("SaveImage", (560, 1500), inputs=[("images", "IMAGE")],
                  widgets=["SDXL_Flux_Outpaint_1080p"], title="save filename", tunable=True)
    G.link(final, "IMAGE", save, "images", "IMAGE")
    preview = G.node("PreviewImage", (560, 1620), inputs=[("images", "IMAGE")], title="preview")
    G.link(final, "IMAGE", preview, "images", "IMAGE")

    # LoadImage stays external (preview/mask-editor); it fans to gis + fit. Proxied onto the
    # box: prompt pair, canvas width/height, SDXL checkpoint, controlnet, Flux repaint
    # denoise, save filename.
    proxy_order = [pos_str, neg_str, tw, th, ckpt, cn_load, ks4, save]
    wires = [("IMAGE", gis, "image"), ("IMAGE", fit, "image")]
    sg = emit_subgraph_with_image_seed(G, "SDXL-Flux Outpaint", proxy_order, img, wires)
    dump_subgraph(sg, "SDXL-Flux-Outpaint-1080p")


def main():
    build_ideogram()
    build_inpaint()
    build_outpaint()
    build_sdxl_flux_outpaint()
    build_video()


if __name__ == "__main__":
    main()
