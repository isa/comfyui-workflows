# LTX-2.3 Keyframe → Video Pipeline (3 meta stages)

This repo has two generations of workflow:

- **4 standalone single-purpose workflows** (below, ["Extracted workflows"](#extracted-standalone-workflows-recommended)) — simpler, decoupled, recommended for most use.
- **The original 3-meta-stage subgraph pipeline** (rest of this doc) — one graph that chains image gen → inpaint → video, useful when you want the frames to feed each other automatically.

## Extracted standalone workflows (recommended)

Four independent, single-purpose workflows. All save **exactly 1920×1080**: they generate
at 1920×1088 (LTX/Flux need mult-of-64 latents) then an `ImageCrop` shaves 4px off the top
and bottom before saving.

All four are packaged as **one collapsed native subgraph box** each, plus — for
Inpaint/Outpaint/Video — one or more real `LoadImage` nodes sitting right next to that box
(see the mask-editor gotcha below for why those aren't hidden inside). Every other node
(loaders, guider/sampler chain, VAE encode/decode, the 1080p crop, preview) lives hidden in
the collapsed box, and a handful of widgets are surfaced on its outside, grouped sensibly:

| File | Purpose | Exposed controls |
|---|---|---|
| `Ideogram-Image-Gen-1080p.json` | Text-to-image (Ideogram 4) | *on the collapsed box:* (positive, negative prompt) / (**image seed** + ON/OFF toggle — optional img2img reference) / (width, height) / (save filename) |
| `Flux-Inpaint-1080p.json` | Inpaint a region of an image | a real **`LoadImage` node** (paint the mask directly on it, mandatory) next to the collapsed box, which exposes (positive, negative prompt) / (**denoise**, default 0.65) / (width, height, save filename) |
| `Flux-Outpaint-1080p.json` | Extend an image's canvas (outpaint) | a real **`LoadImage` node** (mandatory, fit-scaled and centered — any size/aspect) next to the collapsed box, which exposes (positive, negative prompt) / (width, height, save filename) |
| `LTX-2.3-Video-Gen-1080p.json` | Text/keyframe-to-video with audio | 3 real **`LoadImage` nodes** (first/mid/last frame seeds, each optional) next to the collapsed box, which exposes (positive, negative prompt) / (first/mid/last gen ON/OFF toggles) / (duration) / (width, height) / (fps); saves an MP4 via `SaveVideo` |

Notes:
- **"Image seed"** = an input image, not an RNG seed. For image-gen it's an *optional*
  style/composition reference (toggle-gated — off = plain text-to-image); for
  inpaint/outpaint it's *mandatory* (the image you're editing, no toggle); for video it's
  an optional keyframe per lane (also toggle-gated, one toggle per lane, proxied onto the
  box alongside the prompts — the LoadImage nodes themselves stay external, same as
  inpaint/outpaint, for the same mask-editor/preview reason).
- **Why Inpaint/Outpaint's image seed isn't hidden inside the box:** a proxied widget only
  mirrors a `LoadImage`'s plain filename dropdown — it doesn't carry over the preview
  thumbnail + right-click "Open in MaskEditor" the frontend attaches to a real node
  instance. That made the mask editor unreachable when picking an *already-uploaded* file
  (only worked when uploading fresh). Keeping it as a genuine external node, wired into the
  box through a real input socket, fixes that.
- **Inpaint** scales the source to target resolution with aspect preserved (scale-to-cover
  + center-crop) — it will never stretch your image.
- **Inpaint's `denoise` control (default 0.65):** at `denoise=1.0` the masked region enters
  the sampler as pure noise — the model has no anchor on the *original* shape under the
  mask and is free to invent a different one, guided only by your prompt and the
  surrounding pixels. Lowering denoise keeps it closer to the original structure while
  still letting color/detail shift. Raise it toward 1.0 for bigger structural edits, lower
  it further for very subtle ones. Also: diffusion models generally don't respect negation
  well ("do NOT change X") — phrase what you *want* instead ("same turban shape and
  draping, now yellow") for better results. Outpaint deliberately keeps denoise at 1.0,
  unexposed — its padded border has no original content to preserve, so full regeneration
  there is correct, not a bug.
- **Outpaint** never stretches the source, but it does *fit-scale* it when needed: if the
  source is larger than the target in either dimension (e.g. a tall portrait photo into a
  wide landscape target), it's shrunk uniformly (aspect preserved) just enough to fit
  inside the target bounds — never upscaled, and untouched if it already fits. It's then
  centered and padded out to the target resolution, and Flux Fill generates the new border
  area. Pad amounts are computed automatically from source vs. (scaled) target size —
  nothing to configure there. Any source size/aspect ratio works.
- **Outpaint's seed is randomized** (unlike Inpaint's, which stays fixed for
  reproducibility): a large aspect-ratio change (e.g. portrait source into a landscape
  target) can leave 50%+ of the canvas as newly generated content, and at a large fill
  ratio Flux Fill can default to a "safe" hallucination for a given seed — e.g. duplicating
  the one recognizable subject instead of inventing coherent new scenery, confirmed live.
  Randomizing means each run gets a fresh attempt instead of reproducing the same bad
  result every time. If you still get artifacts like this, also try a more descriptive
  prompt (say what should fill the new space, e.g. "empty garden path continuing left and
  right, no additional people") — the more of the canvas is new content, the more the
  prompt matters, and some retries are normal for large-ratio outpaints.
- **Video** keyframe toggles work like the pipeline's: all off → text-to-video; first+last
  on, mid off (default) → first→last interpolation. `stage1 width/height` (LTX's internal
  half-resolution pass) is computed automatically as target/2, not a separate control —
  only ONE width/height pair is exposed.
- ImageGen/Inpaint/Outpaint's subgraph boxes each also have a hidden `PreviewImage` fed by
  the same 1080p-cropped output as `SaveImage`, so you get a live thumbnail without opening
  the box. Video has no equivalent preview node — `SaveVideo` is its only output.
- Generator: `generator/build_extracted_workflows.py` (reuses the same `CONFIG`/node
  framework as the pipeline generator below, plus its own `emit_single_subgraph()` for
  ImageGen's fully-self-contained box, `emit_subgraph_with_image_seed()` for
  Inpaint/Outpaint's box-plus-one-external-LoadImage layout, and
  `emit_subgraph_with_image_seeds()` for Video's box-plus-three-external-LoadImage layout).
  Edit `CONFIG` in `generator/build_workflow.py` and rerun
  `python3 generator/build_extracted_workflows.py` to regenerate all four.
- These are **structural-only verified** (schema lint + JSON validity) — load each in
  ComfyUI and confirm no red/error nodes before relying on it.
- **Known environment gotcha (not a workflow bug):** `Flux-Inpaint`/`Flux-Outpaint` use
  `DualCLIPLoader`, which can throw `RobertaProcessing.__new__() got an unexpected keyword
  argument 'cls'`. Root cause: `tokenizers` 0.23.0 renamed that constructor's `cls` kwarg to
  `cls_token`; if your server has a `tokenizers>=0.23` (even an `rc` pre-release) alongside
  a `transformers` that still calls the old `cls=` form, it breaks. Fix server-side:
  `pip install --force-reinstall "tokenizers<0.23"` (force-reinstall matters if you're
  currently on a pre-release, since a plain install may not register as an "upgrade").

---

## The 3-meta-stage subgraph pipeline

Three sequential subgraph stages, each hiding its complexity behind one box:

```
 [ Global Resolution ] ──► [ Image Generation ] ──► [ Inpainting ] ──► [ Video Generation ]
 (1080p, fps)               3 optional lanes          3 optional lanes    LTX-2.3 + audio
                            pos/neg + optional        pass-through        uses a frame only if
                            reference image           when off            its gen toggle is on
```

Everything optional is gated by a toggle — flip it off and that branch is skipped entirely (lazy
evaluation); the workflow always runs and degrades gracefully.

## Files
| Path | What |
|---|---|
| `ltx23_keyframe_pipeline_subgraphs.json` | **Clean high-level view.** 3 subgraph boxes (Image Generation → Inpainting → Video Generation) + a global-resolution cluster. Load this. |
| `ltx23_keyframe_pipeline.json` | Flat graph (same pipeline, no collapsing) — fallback / "see everything" view. 152 nodes. |
| `generator/build_workflow.py` | Emits both files. Edit `CONFIG`, then `python3 generator/build_workflow.py`. |

## The toggle system (everything optional)

Inside the subgraphs, each optional thing is a `PrimitiveBoolean` toggle driving lazy `Switch` nodes
(the non-selected branch never executes):

| Toggle | Where | Default | Effect when ON |
|---|---|---|---|
| **first / mid / last gen** | Video Generation box | first+last ON, mid OFF | generate that frame and use it as an LTX keyframe. OFF → no generation, video ignores that frame. |
| **first / mid / last ref image** | Image Generation box (per lane) | OFF | use the lane's reference image for color/style/composition (img2img). OFF → plain text-to-image. |
| **first / mid / last inpaint** | Inpainting box (per lane) | OFF | inpaint that frame (Flux-Fill). OFF → the generated image passes through unchanged. |

So: all gen toggles off → text-to-video. first+last on, mid off → first→last interpolation. Flip a
ref toggle → that frame borrows style/composition from a reference image. Flip an inpaint toggle →
that frame is refined before going to video.

## Global resolution

Top-level primitives (one source of truth), **1080p by default**:
`global width` (1920), `global height` (1088), `video stage1 width/height` (= out/2: 960/544),
`global fps` (24). They feed both the image lanes and the video stage. LTX requires mult-of-64 output
(→ mult-of-32 stage-1); values are pre-rounded. To change resolution, edit these (and keep
stage1 = out/2) or change `CONFIG` and regenerate. `num_frames` (default 97, ~4 s @ 24 fps) must
satisfy `(n−1) % 8 == 0`.

## How to use

1. Load `ltx23_keyframe_pipeline_subgraphs.json`.
2. **Global Resolution** (top level): set resolution/fps.
3. Open **Image Generation** → set each lane's *positive prompt*; optionally load a *reference image*
   and flip that lane's **ref** toggle.
4. Open **Inpainting** → for any frame you want refined, paint the mask on that lane's *LoadImage*
   ("… inpaint mask"), set the inpaint prompt, and flip the **inpaint** toggle.
5. Open **Video Generation** → set the *video positive/negative prompt*; flip **gen** toggles for the
   frames you want (default first+last).
6. Run. Output MP4 (with audio) → `output/LTX23_3meta…mp4`.

## Requirements

**ComfyUI** (recent build — `ComfySwitchNode`, typed `PrimitiveInt/Float`, LTX-2.3 / Ideogram-4 /
Flux-Fill nodes are built-in). **Custom nodes**: `ComfyUI-LTXVideo` (Lightricks); optionally
`ComfyUI-VideoHelperSuite`, `ComfyUI-MultiGPU` + `ComfyUI-ParallelAnything` for the 4-GPU setup.

**Models** (already downloaded; filenames in `CONFIG`): LTX — `ltx-2.3-22b-dev`,
`ltx-2.3-22b-distilled-lora-384-1.1`, `ltx-2.3-spatial-upscaler-x2-1.1`, `comfy_gemma_3_12B_it`;
Ideogram 4 — `ideogram4_fp8_scaled`, `ideogram4_unconditional_fp8_scaled`, `qwen3vl_8b_fp8_scaled`,
`flux2-vae`; Flux Fill — `flux1-fill-dev`, `ae`, `clip_l`, `t5xxl_fp16`.

## Caveats (researched)

- **FLF2V is unofficial for LTX-2.3.** Multi-keyframe uses the core `LTXVAddGuide` (frame-index
  conditioning). Strengths (stage1/stage2): first 0.95/1.0, mid 0.5/0.5, last 0.75/0.8 — tunable in
  `CONFIG`. At distilled CFG≈1 the mask is the sole keyframe enforcer (raise strength to lock a frame).
- **Inpaint uses FLUX.1 Fill**, not Ideogram (Ideogram 4 open-weight has no inpaint node).
- **Reference image = img2img** (`BasicScheduler` denoise 0.8 vs `Ideogram4Scheduler` for t2i,
  switched per lane) — gives color/style/composition influence without a separate IP-adapter.
- **Audio is native** to LTX-2.3 (joint audio-video model) — on by default.
- **Loaders are shared within each meta box** (one Ideogram set in Image Gen, one Flux set in
  Inpaint, one LTX set in Video). For per-lane multi-GPU, duplicate loaders per lane (see git history
  / ask).

## Troubleshooting

- **Red/missing nodes** → ComfyUI Manager → *Install Missing Custom Nodes*; update ComfyUI.
- **Frontend load error ("Cannot convert undefined/null to object")** → load the flat file; it's the
  most tolerance-tolerant. Then regenerate.
- **Static / no motion** → lower `last` strength, raise `num_frames`, make first & last differ.
- **CUDA OOM** → drop to 720p (`out_w/out_h` 1280/704, stage1 640/352), reduce `num_frames`, or use fp8.
- **A node widget error** → a few nodes feed both a widget and a linked input (fps, sigmas); delete
  the redundant `widgets_values` entry if your ComfyUI version complains.

## Regenerate
```bash
python3 generator/build_workflow.py          # writes both pipeline JSONs + runs link/subgraph lint
python3 -m json.tool ltx23_keyframe_pipeline_subgraphs.json > /dev/null   # confirm valid

python3 generator/build_extracted_workflows.py   # writes the 4 standalone JSONs (see top of this doc)
```
