# Visual quality gate

Short low-resolution smoke renders prove graph, scheduler, recovery, continuity,
audio, and encoding behavior. They do not prove visual quality.

A visual-quality comparison is valid only when every variant uses the same:

- prompt and negative prompt;
- seed;
- 16:9 composition;
- five-second / 121-frame duration at 24 fps;
- source assets and audio seed;
- delivery crop and scoring rubric.

The active baseline is the stable LTX 2.3 Q4_K_M GGUF graph. Both draft and
selected-final generations must stay on this loader unless the user explicitly
approves a different model experiment. Additional GPUs run independent GGUF
workers for candidate fan-out; the application does not claim that one GGUF
render is split across multiple cards.

Before activating the production profile, retain the media, graph, timings,
ffprobe data, and subject-aware score for:

1. the current GGUF 4+3 draft-size baseline;
2. the same GGUF seed rerendered through 4+3 at final base resolution;
3. GGUF 4+3 plus the engineer-specified one-step refinement, after owner enablement;
4. the complete GGUF LTX refinement → RTX VSR → VHS delivery graph, after owner enablement.

Accept production only when the final path has no material regression in
subject identity, prompt alignment, motion, temporal stability, audio sync, or
encoding integrity. The platform's “20× better” goal refers to the complete
creative and production workflow, not a literal 20× image-quality metric.
