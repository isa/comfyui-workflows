# LTX-2.3 Keyframe → Video Pipeline (3 meta stages)

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
python3 generator/build_workflow.py          # writes both JSONs + runs link/subgraph lint
python3 -m json.tool ltx23_keyframe_pipeline_subgraphs.json > /dev/null   # confirm valid
```
