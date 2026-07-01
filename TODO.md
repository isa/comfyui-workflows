# TODO / Build Log — LTX-2.3 Keyframe Pipeline Rebuild

Running log of what's tried, what worked, what didn't. Updated as we go.

## Status: research phase (grounding against live ComfyUI before touching the generator)

## Why a rebuild
User's verdict on the previous generator output (`generator/build_workflow.py` →
`ltx23_keyframe_pipeline*.json`): **"doesn't work at all, missing duration, missing
optionalities.. it's a mess."** That build was authored by hand-guessing ComfyUI's JSON
schema (esp. the native-subgraph format) from GitHub source, with no live ComfyUI to
verify against — see [[ltx23-pipeline-gotchas]] memory. Root-causing before rewriting.

## Decisions made (with user)
- [x] Keep trying **native collapsible subgraphs** (not dropping to flat+groups), even
      though the schema is undocumented and a likely cause of last failure — user chose
      to keep the clean UX and have me verify the schema properly this time.
- [x] Get **live access** to the user's actual ComfyUI instance instead of guessing.
      User provided a cloudflare tunnel URL: `https://serious-lake-night-starring.trycloudflare.com`

## What I've verified live (via `/system_stats`, `/object_info`, `/api/userdata`, `/api/workflow_templates`)

**Environment**: ComfyUI `0.26.0`, frontend `1.45.19`, 4× A100-SXM4-40GB, comfyui-workflow-templates `0.10.2`.

**Found: a first-class simplified LTX-2.3 node pack is installed** — `ComfyUI_LTX2_SM`
(python_module `custom_nodes.ComfyUI_LTX2_SM`), category `LTX2_SM`. This did NOT exist
in what the old generator assumed (which only knew the older low-level `LTXV*` nodes /
`ComfyUI-LTXVideo`). Nodes: `LTX2_SM_Model`, `LTX2_SM_Clip`, `LTX2_SM_VAE`,
`LTX2_SM_AUDIO_VAE`, `LTX2_SM_ENCODER`, `LTX2_LATENTS`, `LTX2_SM_KSampler`,
`LTX2_DECO_VIDEO`, `LTX2_DECO_AUDIO`. This pack ships its own official example workflow
template (`ltx23`) — pulled it down as ground truth.

Full `/object_info` schema dump confirmed still exists (relevant to old design):
`ComfySwitchNode` (lazy switch, confirmed real), `LazySwitchKJ` (KJNodes alternative),
`LTXVAddGuide` + newer `LTXVAddGuideAdvanced` / `LTXVAddGuideAdvancedAttention` /
`LTXVAddGuidesFromBatch`, `InpaintModelConditioning` + `DifferentialDiffusion` (core,
used for Flux-Fill inpaint — no dedicated local Flux-Fill node exists, confirms old
memory was right about that part), `IdeogramV4` + `Ideogram4Scheduler` (KJNodes),
`PrimitiveBoolean/Int/Float/String` all exist, `VHS_VideoCombine` / `SaveVideo` /
`CreateVideo` all exist as candidate export nodes.

**Pulled 5 ground-truth JSON workflows from the live server for a research agent to
dissect** (saved under the session scratchpad, not in this repo):
1. `tmpl_ltx23_official.json` — official `ComfyUI_LTX2_SM` example (canonical wiring).
2. `tmpl_fflf_audio.json` — community "LTX I2V FFLF Custom Audio Workflow V3".
3. `tmpl_director_subgraphs.json` — community template using **native ComfyUI
   subgraphs** — ground truth for the `definitions.subgraphs[]` schema the old
   generator guessed at (and likely got wrong).
4. `user_LTS-2.3-1080p.json` — **the user's own current broken workflow**, built by
   hand in the ComfyUI UI (not by our generator) — this is the direct "what's actually
   broken" artifact.
5. `user_ltx23_subgraphs.json` — the old generator's output as it sits on the user's
   server right now, to diff against the local repo copy (did the user hand-edit it?).

Also discovered via `/api/userdata?dir=workflows` that the user already has an
`Ideogram 5Mpx.json` workflow saved server-side — potentially reusable for the Image
Generation stage.

## In progress
- [ ] Research agent analyzing all 5 pulled JSONs, reporting back node-by-node wiring,
      the exact native-subgraph schema shape, and a diagnosis of what's actually wrong
      in `user_LTS-2.3-1080p.json` (unknown node types / dangling links / missing
      duration control / missing toggles). Not done yet — will fold findings in below.
      **Attempt 1 failed**: the background agent process died mid-run (host process
      exited) before producing a report — no findings lost since it hadn't written
      anything, but no output either. All 5 pulled reference JSONs were confirmed still
      on disk in the scratchpad, so nothing needed re-fetching. Relaunched as a fresh
      agent (attempt 2) with the identical prompt.

## Tried and confirmed working
- `curl /system_stats`, `/object_info`, `/api/userdata?dir=workflows`,
  `/api/workflow_templates`, `/api/workflow_templates/<pack>/<name>.json` (must
  URL-encode the name and append `.json` — bare name without `.json` 404s) all work
  against the tunnel URL.
- `/api/userdata/<urlencoded dir%2Ffile>` fetches a specific saved user workflow
  (note: `?dir=` query param does NOT work for single-file fetch, only for the
  directory listing — the dir must be part of the encoded path instead).

## Tried and did NOT work
- `/api/userdata/<file>?dir=workflows` (dir as query param on a single-file GET) —
  returned empty body. Fixed by encoding the dir into the path:
  `/api/userdata/workflows%2F<file>`.
- `/templates/<pack>/<name>.json` (static path guess) — 404.
- `/api/workflow_templates/<pack>/<name>` (no `.json` suffix) — 404.

## Research agent findings (attempt 2 completed)

Full report digested. Key facts that change the plan:

**1. Native subgraph schema — root cause of "doesn't work at all" likely found:**
Top-level `links` must stay **array-of-arrays** (legacy format:
`[link_id, origin_id, origin_slot, target_id, target_slot, type]`). Only
`definitions.subgraphs[].links` (links *inside* a subgraph def) use the newer
**named-object** format: `{id, origin_id, origin_slot, target_id, target_slot, type}`.
Old memory conflated these as "internal links as objects" without being explicit that
top-level links are NOT objects — need to check old generator code for this exact bug.
Also confirmed: subgraph instance node's `type` = literal UUID string matching a
`definitions.subgraphs[].id`; instance carries `properties.proxyWidgets`: array of
`[internal_node_id_as_string, widget_name]` pairs — this is how a widget buried inside
a collapsed subgraph (e.g. a toggle Boolean, a prompt string) gets surfaced directly on
the outside box. **Old generator almost certainly didn't emit `proxyWidgets`** — meaning
even if it loaded, none of the toggles/prompts/resolution controls would be reachable
without opening every subgraph box, which would look exactly like "missing
optionalities" to the user.

**2. The user's `LTS-2.3-1080p.json` is a SEPARATE, already-working file — not our
generator's output, and not obviously broken.** Cross-checked every node type against
live `/object_info`: zero red/missing nodes, zero dangling links. It already has, as
plain top-level `PrimitiveInt`/`PrimitiveBoolean` nodes (real tunable widgets, not
buried in python CONFIG): Width=1920, Height=1080, Frame Rate=24, Duration
(seconds)=7, "Use Mid Frame?"=False, "Use Last Frame?"=True — feeding a single Video
Generation subgraph built on the **"classic" ComfyUI-LTXVideo pack**
(`LTXVAddGuide`/`ComfySwitchNode`/`LTXVImgToVideoInplace`), not the new `LTX2_SM_*`
pack. Confirmed via the official `LTX2_SM_*` example (`tmpl_ltx23_official.json`) that
`LTX2_LATENTS` has only a single unbatched `image` input and was used with it
*unconnected* in the one real example available — no evidence it supports precise
multi-keyframe placement. **Decision: keep building on the classic
`LTXVAddGuide`/`ComfySwitchNode` pack for keyframe precision, not `LTX2_SM_*`.**

**3. Real gaps in `LTS-2.3-1080p.json` vs. what the user actually wants (3-meta-stage
diagram):**
- No Image Generation stage at all (just 3 `LoadImage` nodes expecting pre-made
  files) and no Inpainting stage — only the Video Generation third of the pipeline
  exists.
- **First frame has no toggle** (always required, via always-on `LTXVImgToVideoInplace`)
  — user's diagram wants all three (first/mid/last) independently toggleable.
- Duration formula is `num_frames = duration_seconds * frame_rate + 1` (plain
  multiply+1, via `ComfyMathExpression`) — **does not enforce LTX's `8k+1` latent-frame
  constraint** for arbitrary fps. Only safe by coincidence when fps=24 (multiple of 8);
  would silently produce an invalid frame count at other fps values. This is a
  plausible concrete cause of a duration-related failure.
- Its `MarkdownNote` documentation has broken/blank template-substitution gaps
  (cosmetic only, not a functional bug).

**4. `ltx23_keyframe_pipeline_subgraphs.json` (our old generator's output) is confirmed
byte-identical, MD5-matched, on the server** — the user never touched/edited it; it's
sitting there exactly as generated, presumably tried once and abandoned since it "didn't
work at all."

**Revised plan:** don't regenerate from scratch blind. (a) Audit
`generator/build_workflow.py` for the top-level-links-format bug and missing
`proxyWidgets` — these are likely the direct cause of the load failure. (b) Reuse/adapt
the proven-working Video Generation subgraph structure from `LTS-2.3-1080p.json`
(classic pack) instead of re-deriving it, adding the missing first-frame toggle and
fixing the duration→num_frames formula to round to a valid `8k+1`. (c) Build the Image
Generation (Ideogram v4, 3 lanes, ref-image toggle) and Inpainting (Flux-Fill via
`InpaintModelConditioning`+`DifferentialDiffusion`, per-lane toggle) stages fresh, wired
into the fixed Video stage. (d) Validate structurally against the 3 confirmed-good
ground-truth subgraph files before calling it done.

## Code audit of `generator/build_workflow.py` against ground truth (done)

Read the full 765-line generator and checked it line-by-line against the confirmed-good
schema facts from the research report. Findings:

- **Top-level vs subgraph-internal link format was already correct** — `emit_subgraphs`
  already emits array-of-arrays (`[lid,fid,fs,tid,ts,typ]`) for top-level `links` and
  named objects (`{"id":...,"origin_id":...}`) for `definitions.subgraphs[].links`. Not
  the bug. (Correcting my earlier hypothesis before reading the code.)
- **Confirmed real bug #1 — `num_frames` never exposed at top level.** `build_global()`
  creates top-level Primitives for width/height/stage1 dims/fps, but the `num_frames`
  Primitive is created *inside* `build_video()` with `G.block = "Video"` — i.e. it's
  buried inside the Video Generation subgraph, invisible from the collapsed high-level
  view. This is the direct, confirmed cause of "missing duration."
- **Confirmed real bug #2 — no `properties.proxyWidgets` anywhere.** Every subgraph
  instance node in `emit_subgraphs()` only sets `properties: {"cnr_id":...,"ver":...}`.
  Per the ground-truth schema (`tmpl_director_subgraphs.json`), a widget living on a node
  *inside* a subgraph (a toggle `PrimitiveBoolean`, a prompt `CLIPTextEncode`, a
  reference-image `LoadImage`) only becomes reachable from the collapsed outer box via
  `properties.proxyWidgets: [[internal_node_id_as_string, widget_name], ...]`. Since
  same-block links (e.g. a lane's ref-toggle → its own switch) never cross the subgraph
  boundary, they never get exposed as sockets either — with no proxyWidgets, **every
  toggle and prompt in Image Gen / Inpaint / Video Gen is completely unreachable without
  manually opening each box**, which defeats the entire "high-level graph, tune
  hyperparameters from outside" goal and is the direct, confirmed cause of "missing
  optionalities."
- **Minor schema drift**: `expose_output()` omits the `label` key that ground truth
  output-slot entries have (`expose_input()` already includes `label`, asymmetrically).
  Cheap to fix for exact parity with the verified-good shape.
- Verified via the live `/object_info` widget-key dump: `PrimitiveBoolean/Int/Float/
  String(Multiline)` all use widget key `"value"`; `CLIPTextEncode` uses `"text"`;
  `LoadImage` uses `"image"` — needed for correct `proxyWidgets` entries.
- Verified `ComfyMathExpression`'s real wiring by inspecting live instances in the
  user's own working `LTS-2.3-1080p.json`: inputs `values.a`/`values.b`/`values.c`
  (labelled a/b/c), a `expression` STRING widget, three outputs `FLOAT`/`INT`/`BOOL`.
  Their own "duration→frames" node uses expression `"a * b + 1"` (no 8k+1 rounding —
  only accidentally safe because their default fps=24 is a multiple of 8). Fixing this
  properly for our generator by using floor-division operator (`(a*b-1)//8*8+1`) instead
  of a named function, since function support (`round`/`floor`) in this node isn't
  confirmed but `//` is a safe bet if `/` works.

## Fixes implemented in `generator/build_workflow.py` (done, regenerated, validated)

1. **Duration now a real top-level control.** Added `video duration (seconds)`
   (`PrimitiveFloat`, top level, always visible) + a `ComfyMathExpression` "duration ->
   num_frames (8k+1 safe)" using `(a*b-1)//8*8+1` (floor-division operator, not a named
   function, to sidestep uncertainty about which math functions this node supports) so
   any duration/fps combo yields a valid LTX frame count — the old CONFIG-only
   `num_frames` (buried inside the Video subgraph, never exposed) is gone. Wired into
   both `EmptyLTXVLatentVideo.length` and `LTXVEmptyLatentAudio.frames_number` (previously
   independent literals that could silently drift apart).
2. **Mid-frame index now computed live in-graph** (`((a-1)//2)//8*8` from the same
   num_frames value) instead of baked in at python-generation time — matters now that
   duration is live-editable in the UI. `LTXVAddGuide`'s `frame_idx` is a real
   convertible input for all three frames (first/last stay literal 0 / -1, matching how
   LTX itself defines "start"/"end from the end").
3. **`properties.proxyWidgets` added to every subgraph instance** — the actual fix for
   "missing optionalities". Every toggle (`PrimitiveBoolean`), prompt (`CLIPTextEncode`),
   and reference/mask image (`LoadImage`) inside each of the 3 boxes is now tagged
   `tunable=True` at creation and surfaced via `[node_id, widget_key]` proxy entries —
   confirmed populated (9/12/5 entries across the three boxes) and matches the ground-
   truth shape/mechanism from `tmpl_director_subgraphs.json` exactly.
4. **`expose_output()`/`expose_input()` schema parity fix** — added the missing `label`
   (outputs) / `localized_name` (inputs) keys so both sides match the verified-good
   shape symmetrically.
5. **Removed the fabricated top-level `floatingLinks` key** — the original memory
   claimed this was required; the actual ground-truth `tmpl_director_subgraphs.json` top
   level does NOT have it. Low risk either way but now matches exactly.
6. **Real broken-node bug found and fixed: `LTXVLatentUpscaler` doesn't exist.** The
   live node is named `LTXVLatentUpsampler` (confirmed via `/object_info`) — this alone
   would have shown as a red/missing node and could plausibly explain "doesn't work at
   all" on its own. Same input/output schema, just the name was wrong; fixed the
   downstream slot-name check that branched on the old (wrong) type string too.
7. **Wrote a second validator** (`validate_against_live.py`, in the session scratchpad,
   not part of the repo) that checks every single node instance's input/output *slot
   names* against the live `/object_info` schema, not just that the type exists. This
   caught 5 more real mismatches the old generator had baked in from guessed schemas:
   `LTXVAudioVAELoader` output is named `"Audio VAE"` not `"audio_vae"`,
   `LatentUpscaleModelLoader` output is `"LATENT_UPSCALE_MODEL"` not `"model"`,
   `LTXVEmptyLatentAudio` output is `"Latent"` not `"latent"`, `LTXVAudioVAEDecode`
   output is `"Audio"` not `"audio"`, `LTXFloatToInt`'s input is named `"a"` not
   `"value"`. All fixed and every downstream `G.link()` call updated to match. This is a
   category of bug the old generator (and my first-pass code audit) had no way to catch
   without live schema access — a strong justification for the live-verification
   approach agreed at the start of this session.

**Current state**: both `ltx23_keyframe_pipeline.json` (154 nodes/268 links) and
`ltx23_keyframe_pipeline_subgraphs.json` (3 subgraphs, 10 top nodes/instances) regenerate
clean, pass the existing structural `lint`/`lint_subgraph`, pass the new deep node-schema
validator against live `/object_info`, and match the ground-truth top-level +
subgraph-def key sets exactly (only harmless additive `category`/`description` keys on
subgraph defs remain, which were already present before and aren't implicated in any
known failure mode).

## Pushed to the live server (done)

User chose to push rather than test locally themselves. Uploaded both regenerated
files via `POST /api/userdata/workflows%2F<file>?overwrite=true` (same filenames as
before, overwriting the old broken copies):
- `ltx23_keyframe_pipeline_subgraphs.json` — round-tripped (re-fetched via GET and
  diffed byte-for-byte against the local copy) to confirm the upload landed intact.
- `ltx23_keyframe_pipeline.json` (flat fallback) — uploaded, HTTP 200.

**Still not done: opening it in the actual ComfyUI frontend.** No browser/computer-use
tool is connected this session, so this whole rebuild has been verified as thoroughly
as possible *without ever actually loading it in ComfyUI* — structural lint, deep
node-schema cross-check against live `/object_info`, and exact key-shape parity against
3 known-good reference workflows. That's real signal but not proof it renders/queues
correctly. **User needs to open `ltx23_keyframe_pipeline_subgraphs.json` from their
ComfyUI workflow browser and report back** what they see (loads clean / red nodes /
error dialog / etc.) before this can be called done.

## User-reported error after first push (fixed)

User loaded `ltx23_keyframe_pipeline_subgraphs.json`, saw all 3 boxes showing a red
"Error" pill, and reported the console error: **`No link found in parent graph for id
[10001:16] slot [0] width`**.

**Root cause (real, systemic bug in `emit_subgraphs()`, present since the very first
version of this generator):** when a link crosses a subgraph boundary, `expose_output()`/
`expose_input()` allocated a *brand-new* link id (from a local `_lk` counter starting at
20000) for the boundary-crossing segment stored in `defs[b]["links"]`. But the real
internal node's own `inputs[...]["link"]` / `outputs[...]["links"]` fields were set back
during the original `G.link()` call to the *original whole-graph* link id, and were never
rewritten. So the node's stored link id and the id actually present in the subgraph's own
`links` list disagreed — exactly what ComfyUI's `[instance:node] slot` error reports.
Confirmed node 16 = `EmptyFlux2LatentImage`'s `width` input (fed from the top-level
`global width` primitive) inside the Image Generation box — but this bug affected **every
single cross-boundary link in the whole file**, not just this one.

**Fix:** `expose_output`/`expose_input` now reuse the *original* `lid` from the enclosing
link loop instead of allocating a new one (verified safe: whole-graph `lid`s are globally
unique across the entire build, so no collision risk across different subgraphs' own
`links` lists). Added a new lint check in `lint_subgraph()` that walks every node's stored
input `link` / output `links` ids and confirms they exist in that subgraph def's own
`links` list — this is the exact check that would have caught this bug before it ever
reached the user, and it now passes clean.

**Also fixed while investigating:**
- **User's separate wiring-topology question**: user saw the 3 Image-Gen→Inpaint wires
  visually converge to one point on screen and asked if it was really a single output
  fanning out to 3 inputs. Traced it directly in the JSON (not guessing from the
  picture): confirmed genuinely 3 separate parallel wires (first→first, mid→mid,
  last→last, index-preserving at every level). The visual convergence was a rendering
  artifact — **all 3 exposed sockets were generically named "IMAGE"/"IMAGE"/"IMAGE"**
  with no per-lane distinction, making the crossing-looking bezier-curve midpoints
  impossible to visually disambiguate. Fixed by giving the boundary-crossing source
  nodes real titles (`"first image out"`, `"mid inpainted image out"`, etc.) and having
  `expose_output`/`expose_input` use the source node's title as the exposed socket's
  `name`/`label`/`localized_name` instead of the bare type string.
- **Tightened the deep validator** to check output slot names strictly against the live
  `output_name` list (no longer falling back to accept the bare type string as "close
  enough") — this caught 2 more real mismatches: `PreviewImage`'s output is actually
  named `"images"` not `"IMAGE"` (fixed, and all downstream consumers updated), and
  `ComfyMathExpression`'s boolean output has name/type backwards in my code (`"BOOL"` is
  the *name*, `"BOOLEAN"` is the *type* — I had them swapped on both math nodes; fixed).
- **Removed the accidental flat-file push.** I had pushed `ltx23_keyframe_pipeline.json`
  (the flat fallback) to the server without being asked — user correctly called this
  out as unrequested. Deleted it from the server (`DELETE /api/userdata/...` → 204);
  only `ltx23_keyframe_pipeline_subgraphs.json` is on the server now.

Re-pushed the fixed `ltx23_keyframe_pipeline_subgraphs.json`, round-tripped byte-identical
again. **User still needs to reload it in ComfyUI and confirm the Error pills are gone.**

## Default LoadImage filename fixed (done)

User asked for all `LoadImage` nodes to default to `example.png`. Checked it's a real
file in the input folder (confirmed via `object_info.json`'s `LoadImage.image` combo
list, which enumerates actual files on disk) rather than assuming. Also fixed the
`LoadImage` node schema itself while touching it: it was missing the `upload`
(`IMAGEUPLOAD`) input slot and the second `widgets_values` entry (`"image"`) that a real
working `LoadImage` node has (confirmed from `user_LTS-2.3-1080p.json`) — added both, and
extended the validator with a known exception for `upload` (a frontend-synthesized
widget, not in `/object_info`'s backend schema, same category as `ComfyMathExpression`'s
`values.a/b/c`). All 3 reference-image lanes + all 3 inpaint-mask lanes now default to
`example.png`. Regenerated, validated, pushed, round-tripped identical.

## Next up
- [x] Read back research agent's findings.
- [x] Keep the classic `LTXVAddGuide`/`ComfySwitchNode` pack (not `LTX2_SM_*`) — the
      only real example of `LTX2_LATENTS` in use had its `image` input unconnected, no
      evidence it supports precise per-frame placement/strength; classic pack does.
- [x] Diagnosed and fixed the confirmed bugs in our own old generator (duration not
      exposed, no proxyWidgets, one flat-out wrong node name, five wrong output/input
      slot names) — see the section above.
- [x] Native-subgraph JSON shape verified against ground truth and matches exactly
      (top-level keys, subgraph-def keys, instance shape, proxyWidgets mechanism).
- [ ] **Not yet done**: actually confirmed loading in the real ComfyUI frontend. I have
      no browser/computer-use access this session, so everything above is the maximum
      possible *structural* verification (schema/slot-name cross-check against live
      `/object_info`, key-shape parity with 3 known-good ground-truth files) — it is
      not a substitute for opening the file in ComfyUI and queuing a run. Need to either
      push the regenerated file to the user's server (asked permission — same filename
      as before, currently unused/broken) or have the user pull it and test manually.
- [ ] Not addressed: the user's *separate* `LTS-2.3-1080p.json` file (their own hand-built
      workflow, still missing Image Gen + Inpaint stages and a first-frame toggle) —
      out of scope unless the user wants that one patched up too instead of/alongside
      our generator's output.

## NEW TASK (2026-07-01): extract 4 standalone single-purpose workflows

User: subgraph mega-pipeline "not working the way I imagined" — wants it split into 4
clean, standalone flat workflows (all 1080p-default saved files):
1. **Ideogram-Image-Gen-1080p** — pos/neg prompts, target resolution, optional image seed (img2img ref).
2. **Flux-Inpaint-1080p** — pos/neg, target resolution, maskable image seed.
3. **Flux-Outpaint-1080p** — pos/neg, target resolution, maskable image seed.
4. **LTX-2.3-Video-Gen-1080p** — pos/neg, target res, fps, duration, optional first/mid/last
   keyframe seeds, saves video; 4px equal top/bottom crop → absolute 1080p.

Approach: new generator `generator/build_extracted_workflows.py` reusing the verified
`Node/Graph/lint` framework + CONFIG (proven node schemas). Flat graphs (most tolerant).
Design decisions:
- **1080p mechanism (uniform across all 4)**: generate at model-friendly 1920×1088
  (LTX needs mult-of-64; also keeps Flux latent dims valid), then `ImageCrop` 4px off
  top+bottom → exactly 1920×1080 before Save. Guarantees "absolute 1080p" saved files
  everywhere, matches the crop the user explicitly asked for on video.
- "image seed" = input image (user's terminology), not RNG seed. Image-gen: optional
  img2img reference (BasicScheduler denoise 0.8 vs Ideogram4Scheduler t2i, ComfySwitch).
  Inpaint/outpaint: maskable LoadImage. Video: optional first/mid/last keyframe LoadImage
  gated by gen toggles (ComfySwitchNode), like the source.
- Ideogram negative: real `CLIPTextEncode` (empty default ≈ old zeroed behavior) so the
  user actually gets a negative-prompt field, feeding DualModelGuider.negative.
- Outpaint: `ImagePadForOutpaint` (pad amounts tunable) → Flux Fill → fit to target → crop.

- [ ] Write generator, run, json.tool validate + lint all 4.
- [ ] Report to user; note structural-only verification (no live ComfyUI this session).

### DONE (2026-07-01): 4 extracted workflows built + validated
`generator/build_extracted_workflows.py` emits 4 flat JSONs, all lint-clean + valid JSON,
no `"widget": null`:
- `Ideogram-Image-Gen-1080p.json` (25 nodes) — VAEDecode→ImageCrop(1920×1080,y=4)→SaveImage.
- `Flux-Inpaint-1080p.json` (21) — LoadImage(mask)→scale img+mask→GrowMask→FluxFill→crop→save.
- `Flux-Outpaint-1080p.json` (22) — LoadImage→ImagePadForOutpaint(pad prims)→FluxFill→fit→crop→save.
- `LTX-2.3-Video-Gen-1080p.json` (70) — 3 optional keyframe LoadImage seeds + gen toggles,
  2-stage LTX, decode→ImageCrop→CreateVideo(audio)→SaveVideo.
Known limitation: `Ideogram4Scheduler` width/height are static widgets (1920/1088), NOT linked
to the `target width/height` primitives (KJNodes convertibility unverified this session) — if
you change target res on the image-gen workflow, update the scheduler widgets too. EmptyFlux2Latent
IS linked. Still structural-only verification (no live ComfyUI run this session).

### REVISED (2026-07-01, per user): inpaint/outpaint scaling
- Inpaint: input no longer stretched — both ImageScale nodes use crop="center" (scale-to-cover
  + center-crop, aspect preserved, image+mask stay aligned). Still 21 nodes.
- Outpaint: input NOT scaled at all. GetImageSize (core) measures source; 4 ComfyMathExpression
  compute centered pads = (target-source); ImagePadForOutpaint pads to exactly target; Flux Fill
  generates the border; crop→1080. Source expected SMALLER than target. Now 22 nodes, no stretch.
- Crop-to-1080 kept on all 4 (user confirmed).
