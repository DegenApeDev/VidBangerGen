# Architecture

## Domain model

- A **project** owns the creative brief and lifecycle.
- A **concept** is one creative treatment; exactly one is selected for production.
- A **shot** is an independently renderable beat with a target duration and continuity position.
- A **candidate** is one seed/settings/output combination for a shot.
- A **job** is a persistent unit of work in either the `comfy` or `local` lane.
- An **asset** is an uploaded reference image/video, visual bible, music bed, or voiceover.
- An **export** is a finished platform master.
- **Feedback** connects human preference to candidates for later optimization.
- A **chain** preserves the original continue/merge workflow in the same durable store.

SQLite is authoritative. In-memory worker state is observational only. On service startup, interrupted jobs below their retry limit return to the queue.

## Worker topology

Each configured ComfyUI URL creates one generation worker. Workers atomically
claim the oldest queued `comfy` job. This provides candidate-level and
shot-level parallelism across any configured pool. Drafts and selected finals
both stay on GGUF. Dormant DisTorch2 FP8 support is fail-closed, never selected
implicitly, and is not part of the current production path.

The local worker handles concept planning, media scoring, and final exports. Comfy sampler updates arrive through its WebSocket; Server-Sent Events carry persistent job snapshots to the browser.

## Prompt authorship

Every project records `prompt_mode` in its creative brief:

- `assisted` sends the structured brief to the local `bonsai-27b:latest` Q1
  director, normalizes its storyboard, then creates controlled prompt variants
  for Best-of-N exploration;
- `manual` stores the operator's storyboard synchronously, never calls the
  Ollama director, never enriches the prompt, and varies candidate seeds without
  appending take directions.

In either mode, the workflow adapter writes the persisted candidate prompt to
both the Gemma API and CLIP positive-text nodes. The `qwen3-vl:8b` scorer is a
separate post-render judge and cannot mutate generation prompts.

## Continuity

After each candidate finishes, VidBangerGen extracts:

- a final 17-frame H.264 motion tail;
- a final-frame PNG fallback.

The next shot waits until the prior shot is scored and selected. The adapter now builds a full 17-frame `LoadVideo → GetVideoComponents → LTXVAddGuide` path and retains the PNG as a fallback. Upload-conditioned jobs are pinned to the authorized worker whose disposable input directory is writable. Metadata always records which continuity mode actually ran.

## Inference-worker boundary

ComfyUI is an inference appliance, not application-owned infrastructure. The
API uses ComfyUI's HTTP/WebSocket surface to upload temporary conditioning
media, queue in-memory workflow graphs, monitor jobs, and download outputs. The
only SSH operation in the application is a read-only `cat` of the shared API
workflow. VidBangerGen does not write the shared graph, install packages or
models, restart the owner's worker, or use another account. Shared capability
changes remain unavailable until the owner/operator performs and verifies them.

## Reference audio

Reference audio is trimmed to its configured seed interval, silence-padded to the complete generation duration, encoded by `LTXVAudioVAEEncode`, and combined with the video latent. `LTXVSetAudioVideoMaskByTime` preserves the initial seed and applies noise only from the seed boundary through the remainder. The second sampler carries the generated audio forward, and every artifact records the exact preserved/generated intervals.

## Safety and reproducibility

- Prompts, both sampler seeds, frames, FPS, dimensions, workflow version, and profile are persisted.
- Every output prefix contains project and candidate IDs.
- Workflow class types are validated against the manifest before queueing.
- Upload size, names, fields, resolution, duration, and strengths are validated.
- Failed optional creative models use named fallbacks rather than failing generation.
- Production profiles fail closed when their required rig capabilities are absent.
- Cancellation is prompt-scoped on workers marked interrupt-capable. On shared
  workers, the application never sends ComfyUI's global interrupt; it safely
  waits for its own prompt and discards the result.
- Manual winner locks are authoritative and cannot be overwritten by later automatic scores.
- Platform exports apply transitions, crop, captions, and the final audio mix in one CRF 16 video encode instead of compounding two lossy passes.
