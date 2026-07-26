# LTX 2.3 Creative Lab integration

VidBangerGen tracks the official Lightricks Creative Lab as an opt-in capability
catalog. Catalogued does not mean installed, enabled, or quality-accepted. The
API exposes the current state at `GET /creative-lab` and fails closed when a
required capability is absent.

The catalog mirrors the 13 models currently listed by Lightricks: Ingredients,
In/Outpainting, Pixel Spatial Upscaler, Water Simulation, Deblur, Decompression,
Cross-Eyed, Instant Shave, Colorization, Day to Night, Cinemagraph, Foley V2A,
and Clean Plate. LipDub remains visible as an LTX 2.3 companion workflow rather
than being incorrectly presented as a current Creative Lab collection member.
The expected LoRA files must be installed under the configured worker's model
directory. Ingredients, Pixel Spatial, Water Simulation, Day to Night, Deblur,
Decompression, Cross-Eyed, Instant Shave, Colorization, Cinemagraph, Foley V2A,
Clean Plate, and In/Outpainting have isolated GGUF execution paths. The LipDub
companion also has an isolated, two-stage GGUF path. No mode silently falls
back to the Union IC-LoRA or FP8 merely because its weight is present.

## Ingredients visual bible

Projects can store a `reference_sheet` image and choose **Ingredients visual
bible** in the Generation Lab. This is a dedicated Ingredients path rather than
ordinary Union Control conditioning. It:

1. decodes and fits the sheet onto a black 768×448 canvas without stretching or
   cropping it;
2. repeats the still for all 121 frames at 24 fps;
3. compiles the documented `Reference sheet:` and `Generated video:` sections;
4. layers the installed distilled 384 LoRA and Ingredients IC-LoRA over the dev
   Q4_K_M GGUF using the official eight-sigma distilled recipe;
5. disconnects generated audio and locally trims the result to the requested
   story beat (up to five seconds);
6. persists the output as a normal reviewable/selectable production candidate.

All Ingredients takes are currently constrained to the documented landscape
training bucket. Longer stories should use multiple five-second storyboard
beats and the existing chaining/export tools. The operator still judges actual
character, product, logo, wardrobe, and location fidelity before selection.

## Day-to-Night relighting

Day to Night is the first active specialized relighting branch. It runs from an
approved source clip in an isolated ComfyUI API graph using the distilled 1.1
Q4_K_M GGUF and the installed Day-to-Night IC-LoRA. The source clip guides the
full visual pass; generated audio is disconnected and any existing source audio
is restored locally. The current safe route is limited to 121 frames per pass.

## Restoration and clean-plate tools

Candidate cards expose an optional clip-tool selector for Deblur,
Decompression, Colorization, and Clean Plate. They share the identity-safe
stage-one video-guide graph while compiling their distinct trained prompt
conventions (`DEBLUR`, `ENHANCE QUALITY`, `COLORIZE`, or a positive description
of the empty result). Each pass retains the approved source and creates a
separate downloadable result.

Deblur, Decompression, and Colorization restore the untouched source audio
locally after the visual pass. Clean Plate intentionally remains structurally
silent: restoring the source soundtrack could leave removed people speaking or
moving objects audible off-screen. Current 24 GB routes are capped at a
768-pixel long edge and 121 frames per isolated pass.

## Water and character effects

Water Simulation runs a dry 24 fps, `8n+1`-frame source through the documented
eight-step distilled stage-one recipe with CFG 1 and the `ADD WATER` prompt
contract. It requires the operator to describe the fluid, motion, surfaces,
and interactions. The dry soundtrack is discarded because it cannot contain
the added water. The completed Water result can be previewed in the candidate
card and passed directly to Foley V2A for synchronized replacement sound.

Instant Shave and Cross-Eyed use the dev Q4_K_M GGUF with 30 steps, CFG 4, and
STG on block 29. Instant Shave injects `REMOVEBEARD`; Cross-Eyed uses explicit
convergent-eye direction plus normal/corrected-eye negatives. Both are opt-in,
source-video transformations, preserve existing source audio locally, and are
capped at 121 frames on the current route.

## Cinemagraph maker

Cinemagraph is a dedicated still-image-to-video tool in the Enhancement Lab,
not a clip filter. It center-crops the selected image to 512×704 portrait or
704×512 landscape, generates 25 frames at 25 fps with the dev Q4_K_M GGUF,
30 steps, CFG 4, and STG block 29, and adds the required
`CINEMAGRAPH_MOTION` trigger. The prompt must name exactly what moves; the
camera, people, background, and everything else are compiled as frozen. The
delivered loop is structurally silent and remains available as a durable job
download.

## Creative spatial upscaler

The official 2x/4x Pixel Spatial Upscaler is a separate LTX video-to-video
render, not the latent upsampler already used by the 4+3 graph. It synthesizes
new detail, so the platform must label it as a creative rerender and retain the
original approved clip. It will be evaluated as an optional finishing branch,
not as proof of the still-unverified one-step refinement stage.

The official distilled Pixel Spatial example loads the dev checkpoint plus its
distilled LoRA. VidBangerGen now mirrors that stack with the dev Q4_K_M GGUF,
the distilled 384 LoRA, and the selected 2×/4× IC-LoRA in an isolated API graph.
It accepts an uploaded approved clip, freezes existing source audio, saves a
recoverable H.264 MP4, and never mutates the stable generation graph. The 24 GB
route caps output to a 1536-pixel long edge and 121 frames per inference pass.
Longer LTX clips are split into overlapping windows, blended locally, and
remuxed with their original audio. A live x2 preflight completed on the
disposable 8189 worker at 1536×832 without OOM.

No Creative Lab model or workflow is automatically downloaded to a worker.
All observed weights were supplied by the authorized operator; VidBangerGen's
inventory check is read-only and never moves, installs, or deletes model files.

## Video Inpainting and Outpainting

Each candidate card exposes the official two-stage masked-edit route. Inpaint
accepts a static image mask or frame-matched animated mask; white pixels are
regenerated and black pixels remain protected. Outpaint creates its padded
canvas and exact binary mask locally for left, right, top, bottom, or all-side
expansion. The model generates a coarse half-resolution edit, blends it with
the protected source, performs the second boundary-refinement pass, and applies
the final Laplacian blend. Existing source audio is restored locally.

The current route accepts `8n+1` clips up to 121 frames and a 1024-pixel final
edge. Static masks must match source pixel dimensions; animated masks must also
match the source frame count. Prompts describe only the new/masked region, and
removal masks should include contact shadows, reflections, and boundary areas.

## LipDub companion workflow

LipDub is intentionally separate from normal T2V/I2V audio. It accepts an
approved source video containing one visible speaker and that speaker's voice,
plus exact operator-authored replacement dialogue and its language. The source
audio is encoded only as speaker-identity reference tokens. Stage one generates
new lip motion and dialogue audio; stage two performs the spatial upscale while
freezing the generated audio. The dev Q4_K_M GGUF remains the base model.

The current 24 GB route normalizes to 24 fps, 768×448 landscape or 448×768
portrait, and 17–121 `8n+1` frames. Enter the final words in the target
language's native script—the model does not translate—and keep the line close
to the original speech timing for the most reliable result.

## Voice and dialogue policy

LTX 2.3 generates audio and video jointly, so unstructured prose in a video
prompt can be interpreted as speech. VidBangerGen separates four operator
choices before inference:

1. `ambient` is the Studio, Quick, and continuation default. It strips generated
   speech directions, adds an explicit
   zero-human-voice contract, and adds speech/gibberish negatives, but remains
   probabilistic because LTX creates audio and video jointly;
2. `silent` disconnects decoded audio from `CreateVideo`, producing a
   structurally audio-free MP4 and remains the guaranteed no-voice choice;
3. `native-dialogue` removes quoted speech from the visual prose, accepts the
   spoken line through a separate field, and compiles one speaker,
   language/accent, and quoted line into LTX's native flowing format, with a
   conservative 2.5-word/second guard. It is generative, not a verbatim speech
   guarantee;
4. `prompt` leaves a human-authored prompt untouched for advanced use.

The active Foley V2A finisher is an isolated alternative for approved clips.
It freezes source video at a zero noise mask, denoises only an empty audio
latent with the dev Q4_K_M GGUF and Foley LoRA, downloads lossless audio, and
muxes it onto the untouched local video stream. It is limited to 121 frames per
24 GB pass, suppresses speech/music, replaces any source soundtrack, and
supports multiple seed takes because sound synchronization remains generative.
Every completed Creative Lab visual pass is now a valid source for a later
Foley or upscale pass, so processing does not require a download/re-upload
round trip.

Narration and advertising copy is preserved as `voiceover_text`, not repeated
inside every visual prompt. The reliable long-form path is to generate the
visuals silently, upload a recorded or TTS voiceover, and mix that exact
track during local export. This also keeps voice identity stable across shots.

The LipDub IC-LoRA is enabled only from a candidate's explicit **LipDub** tool.
It is a video-to-video speech-replacement workflow and is never loaded into
ordinary T2V/I2V generation.
