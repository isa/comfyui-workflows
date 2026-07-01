#!/usr/bin/env python3
"""
Extract 4 standalone single-purpose FLAT workflows from the 3-meta keyframe pipeline.

  1. Ideogram-Image-Gen-1080p  — t2i (+ optional img2img "image seed"), pos/neg, target res.
  2. Flux-Inpaint-1080p        — Flux Fill inpaint, pos/neg, target res, maskable image seed.
  3. Flux-Outpaint-1080p       — Flux Fill outpaint (ImagePadForOutpaint), pos/neg, target res.
  4. LTX-2.3-Video-Gen-1080p   — LTX-2.3 two-stage + audio, pos/neg, res, fps, duration,
                                 optional first/mid/last keyframe seeds, saves video.

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
from build_workflow import Node, Graph, CONFIG, SIGMAS_S1, SIGMAS_S2, lint  # noqa: E402

C_ID = CONFIG["ideogram"]
C_FF = CONFIG["flux_fill"]
C_LTX = CONFIG["ltx"]
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


def flux_fill_loaders(G, x=-1500):
    unet = G.node("UNETLoader", (x, 0), outputs=[("MODEL", "MODEL")],
                  widgets=[C_FF["diffusion"], "fp8_e4m3fn"])
    clip = G.node("DualCLIPLoader", (x, 180), outputs=[("CLIP", "CLIP")],
                  widgets=[C_FF["clip_l"], C_FF["t5xxl"], "flux"])
    vae = G.node("VAELoader", (x, 360), outputs=[("VAE", "VAE")], widgets=[C_FF["vae"]])
    diff = G.node("DifferentialDiffusion", (x, 540), inputs=[("model", "MODEL")],
                  outputs=[("MODEL", "MODEL")])
    G.link(unet, "MODEL", diff, "model", "MODEL")
    return {"unet": diff, "clip": clip, "vae": vae}


def flux_fill_body(G, flux, tw, th, pixels_src, pixels_slot, mask_src, mask_slot, x=-300):
    """positive/negative prompts + FluxGuidance + InpaintModelConditioning + KSampler + decode.
    Returns the VAEDecode node (IMAGE)."""
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
                   widgets=[SEED, "fixed", C_FF["steps"], C_FF["cfg"], "euler", "normal", 1.0])
    G.link(flux["unet"], "MODEL", ksamp, "model", "MODEL")
    G.link(inp, "positive", ksamp, "positive", "CONDITIONING")
    G.link(inp, "negative", ksamp, "negative", "CONDITIONING")
    G.link(inp, "latent", ksamp, "latent_image", "LATENT")
    dec = G.node("VAEDecode", (x, 740), inputs=[("samples", "LATENT"), ("vae", "VAE")],
                 outputs=[("IMAGE", "IMAGE")])
    G.link(ksamp, "LATENT", dec, "samples", "LATENT")
    G.link(flux["vae"], "VAE", dec, "vae", "VAE")
    return dec


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


# --------------------------------------------------------------------------------------
# 1. Ideogram-Image-Gen-1080p
# --------------------------------------------------------------------------------------

def build_ideogram():
    G = Graph()
    tw, th = res_primitives(G)

    unet = G.node("UNETLoader", (-1500, 0), outputs=[("MODEL", "MODEL")],
                  widgets=[C_ID["diffusion"], "default"])
    unet_neg = G.node("UNETLoader", (-1500, 180), outputs=[("MODEL", "MODEL")],
                      widgets=[C_ID["unconditional"], "default"])
    clip = G.node("CLIPLoader", (-1500, 360), outputs=[("CLIP", "CLIP")],
                  widgets=[C_ID["clip"], "ideogram4", "default"])
    vae = G.node("VAELoader", (-1500, 540), outputs=[("VAE", "VAE")], widgets=[C_ID["vae"]])

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

    # optional "image seed" = img2img reference
    ref_img = G.node("LoadImage", (-1320, 500),
                     inputs=[("image", "COMBO", "image"), ("upload", "IMAGEUPLOAD", "upload")],
                     outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")], widgets=["example.png", "image"],
                     title="image seed (optional img2img ref)", tunable=True)
    ref_lat = G.node("VAEEncode", (-1000, 640), inputs=[("pixels", "IMAGE"), ("vae", "VAE")],
                     outputs=[("LATENT", "LATENT")])
    G.link(ref_img, "IMAGE", ref_lat, "pixels", "IMAGE")
    G.link(vae, "VAE", ref_lat, "vae", "VAE")
    toggle = G.node("PrimitiveBoolean", (-1320, 760), outputs=[("BOOLEAN", "BOOLEAN")],
                    widgets=[False], title="use image seed ON/OFF", tunable=True)
    lat_sw = G.node("ComfySwitchNode", (-1000, 780),
                    inputs=[("switch", "BOOLEAN", "switch"), ("on_true", "LATENT"), ("on_false", "LATENT")],
                    outputs=[("output", "LATENT")], widgets=[False])
    G.link(toggle, "BOOLEAN", lat_sw, "switch", "BOOLEAN")
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
    G.link(toggle, "BOOLEAN", sig_sw, "switch", "BOOLEAN")
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
                  widgets=["Ideogram_1080p"], title="save 1080p")
    G.link(crop, "IMAGE", save, "images", "IMAGE")

    dump(G, "Ideogram-Image-Gen-1080p")


# --------------------------------------------------------------------------------------
# 2. Flux-Inpaint-1080p
# --------------------------------------------------------------------------------------

def build_inpaint():
    G = Graph()
    tw, th = res_primitives(G)
    flux = flux_fill_loaders(G)

    img = G.node("LoadImage", (-1000, 0),
                 inputs=[("image", "COMBO", "image"), ("upload", "IMAGEUPLOAD", "upload")],
                 outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")], widgets=["example.png", "image"],
                 title="image seed (paint mask here)", tunable=True)
    # scale image + mask to target working resolution (mask via image round-trip)
    # crop="center" = scale-to-cover + center-crop (preserves aspect, no stretch). Applied
    # identically to image and mask so they stay pixel-aligned.
    img_s = G.node("ImageScale", (-680, 0),
                   inputs=[("image", "IMAGE"), ("width", "INT", "width"), ("height", "INT", "height")],
                   outputs=[("IMAGE", "IMAGE")], widgets=["lanczos", OUT_W, OUT_H, "center"])
    G.link(img, "IMAGE", img_s, "image", "IMAGE")
    G.link(tw, "INT", img_s, "width", "INT")
    G.link(th, "INT", img_s, "height", "INT")
    m2i = G.node("MaskToImage", (-680, 160), inputs=[("mask", "MASK")], outputs=[("IMAGE", "IMAGE")])
    G.link(img, "MASK", m2i, "mask", "MASK")
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

    dec = flux_fill_body(G, flux, tw, th, img_s, "IMAGE", grow, "MASK", x=-300)
    crop = crop_to_1080(G, tw, th, dec, "IMAGE", (20, 900))
    save = G.node("SaveImage", (-300, 920), inputs=[("images", "IMAGE")],
                  widgets=["Flux_Inpaint_1080p"], title="save 1080p")
    G.link(crop, "IMAGE", save, "images", "IMAGE")

    dump(G, "Flux-Inpaint-1080p")


# --------------------------------------------------------------------------------------
# 3. Flux-Outpaint-1080p
# --------------------------------------------------------------------------------------

def build_outpaint():
    G = Graph()
    tw, th = res_primitives(G)
    flux = flux_fill_loaders(G)

    img = G.node("LoadImage", (-1000, 0),
                 inputs=[("image", "COMBO", "image"), ("upload", "IMAGEUPLOAD", "upload")],
                 outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")], widgets=["example.png", "image"],
                 title="image seed (to extend)", tunable=True)
    # No stretch: keep the source at native size, measure it, and pad (centered) so the
    # padded canvas == target resolution. Flux Fill then generates the surrounding border.
    # (Source is expected SMALLER than target — that's the outpaint use case.)
    gis = G.node("GetImageSize", (-680, -160), inputs=[("image", "IMAGE")],
                 outputs=[("width", "INT"), ("height", "INT"), ("batch_size", "INT")])
    G.link(img, "IMAGE", gis, "image", "IMAGE")
    left = math_expr(G, (-680, 0), "(a-b)//2", tw, "INT", "pad left = (target_w - w)//2",
                     b_src=gis, b_slot="width")
    right = math_expr(G, (-680, 140), "(a-b)-((a-b)//2)", tw, "INT", "pad right", b_src=gis, b_slot="width")
    top = math_expr(G, (-680, 280), "(a-b)//2", th, "INT", "pad top = (target_h - h)//2",
                    b_src=gis, b_slot="height")
    bottom = math_expr(G, (-680, 420), "(a-b)-((a-b)//2)", th, "INT", "pad bottom", b_src=gis, b_slot="height")
    pad = G.node("ImagePadForOutpaint", (-340, 0),
                 inputs=[("image", "IMAGE"), ("left", "INT", "left"), ("top", "INT", "top"),
                         ("right", "INT", "right"), ("bottom", "INT", "bottom")],
                 outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")], widgets=[0, 0, 0, 0, 40])
    G.link(img, "IMAGE", pad, "image", "IMAGE")
    G.link(left, "INT", pad, "left", "INT")
    G.link(top, "INT", pad, "top", "INT")
    G.link(right, "INT", pad, "right", "INT")
    G.link(bottom, "INT", pad, "bottom", "INT")

    # padded canvas is already target resolution -> straight to Flux Fill, then crop to 1080
    dec = flux_fill_body(G, flux, tw, th, pad, "IMAGE", pad, "MASK", x=-300 + 320)
    crop = crop_to_1080(G, tw, th, dec, "IMAGE", (340, 900))
    save = G.node("SaveImage", (20, 920), inputs=[("images", "IMAGE")],
                  widgets=["Flux_Outpaint_1080p"], title="save 1080p")
    G.link(crop, "IMAGE", save, "images", "IMAGE")

    dump(G, "Flux-Outpaint-1080p")


# --------------------------------------------------------------------------------------
# 4. LTX-2.3-Video-Gen-1080p
# --------------------------------------------------------------------------------------

def _guide_stage(G, stage, x, y0, cond, base_lat, frames, toggles, ltx):
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
        fi = guide.add_input("frame_idx", "INT", widget="frame_idx")
        if not isinstance(idx, int):
            G.link(idx, "INT", guide, fi, "INT")
        G.link(cur_pos, pos_slot, guide, "positive", "CONDITIONING")
        G.link(cur_neg, neg_slot, guide, "negative", "CONDITIONING")
        G.link(ltx["vae"], "VAE", guide, "vae", "VAE")
        G.link(cur_lat, lat_slot, guide, "latent", "LATENT")
        G.link(img, 0, guide, "image", "IMAGE")  # LoadImage IMAGE slot
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


def build_video():
    G = Graph()
    # resolution + fps + duration primitives
    tw = G.node("PrimitiveInt", (-2200, 0), outputs=[("INT", "INT")], widgets=[OUT_W],
                title="target width", tunable=True)
    th = G.node("PrimitiveInt", (-2200, 120), outputs=[("INT", "INT")], widgets=[OUT_H],
                title="target height", tunable=True)
    s1w = G.node("PrimitiveInt", (-2200, 240), outputs=[("INT", "INT")], widgets=[S1_W],
                 title="video stage1 width (= out/2)", tunable=True)
    s1h = G.node("PrimitiveInt", (-2200, 360), outputs=[("INT", "INT")], widgets=[S1_H],
                 title="video stage1 height (= out/2)", tunable=True)
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

    # optional keyframe "seeds" (LoadImage) — used only if the frame's gen toggle is ON
    first_img = G.node("LoadImage", (x - 700, 0),
                       inputs=[("image", "COMBO", "image"), ("upload", "IMAGEUPLOAD", "upload")],
                       outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")], widgets=["example.png", "image"],
                       title="first frame seed (optional)", tunable=True)
    mid_img = G.node("LoadImage", (x - 700, 1480),
                     inputs=[("image", "COMBO", "image"), ("upload", "IMAGEUPLOAD", "upload")],
                     outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")], widgets=["example.png", "image"],
                     title="mid frame seed (optional)", tunable=True)
    last_img = G.node("LoadImage", (x - 700, 2960),
                      inputs=[("image", "COMBO", "image"), ("upload", "IMAGEUPLOAD", "upload")],
                      outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")], widgets=["example.png", "image"],
                      title="last frame seed (optional)", tunable=True)

    frames = {
        "first": (first_img, 0, CONFIG["strength_first"][0], CONFIG["strength_first"][1], CONFIG["gen_first"]),
        "mid": (mid_img, mid_idx, CONFIG["strength_mid"][0], CONFIG["strength_mid"][1], CONFIG["gen_mid"]),
        "last": (last_img, -1, CONFIG["strength_last"][0], CONFIG["strength_last"][1], CONFIG["gen_last"]),
    }
    t_first = G.node("PrimitiveBoolean", (x - 320, 0), outputs=[("BOOLEAN", "BOOLEAN")],
                     widgets=[CONFIG["gen_first"]], title="use first frame ON/OFF", tunable=True)
    t_mid = G.node("PrimitiveBoolean", (x - 320, 1480), outputs=[("BOOLEAN", "BOOLEAN")],
                   widgets=[CONFIG["gen_mid"]], title="use mid frame ON/OFF", tunable=True)
    t_last = G.node("PrimitiveBoolean", (x - 320, 2200), outputs=[("BOOLEAN", "BOOLEAN")],
                    widgets=[CONFIG["gen_last"]], title="use last frame ON/OFF", tunable=True)
    toggles = {"first": t_first, "mid": t_mid, "last": t_last}

    ltx = {"vae": ckpt, "lora": lora, "audio_vae": audio_vae, "upscaler": upscaler}
    s1_pos, s1_neg, s1_lat = _guide_stage(G, "stage1", x + 500, 0, cond, vid_lat, frames, toggles, ltx)

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
    noise1 = G.node("RandomNoise", (x + 1100, 420), outputs=[("NOISE", "NOISE")], widgets=[SEED, "fixed"])
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

    up = G.node("LTXVLatentUpsampler", (x + 1500, 0),
                inputs=[("samples", "LATENT"), ("upscale_model", "LATENT_UPSCALE_MODEL"), ("vae", "VAE")],
                outputs=[("LATENT", "LATENT")])
    G.link(sep1, "video_latent", up, "samples", "LATENT")
    G.link(upscaler, "LATENT_UPSCALE_MODEL", up, "upscale_model", "LATENT_UPSCALE_MODEL")
    G.link(ckpt, "VAE", up, "vae", "VAE")

    s2_pos, s2_neg, s2_lat = _guide_stage(G, "stage2", x + 1900, 0, cond, up, frames, toggles, ltx)
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
    noise2 = G.node("RandomNoise", (x + 2400, 420), outputs=[("NOISE", "NOISE")], widgets=[SEED + 1, "fixed"])
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

    dump(G, "LTX-2.3-Video-Gen-1080p")


def main():
    build_ideogram()
    build_inpaint()
    build_outpaint()
    build_video()


if __name__ == "__main__":
    main()
