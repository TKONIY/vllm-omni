# LTX2 CFG-Parallel Parity Artifacts

Original GIF assets:
- `old_vs_new.gif`: left is the preref baseline output; right is the refactored LTX2 output with post-step `torch.cuda.current_stream(...).synchronize()`.
- `diff.gif`: per-frame visual difference between the two videos. It is blank because the outputs are bit-identical.
- `MD5(/tmp/ltx2_preref_small.mp4) = 5bc66ef7d4dbc0074195a1d7ed01a4ef`
- `MD5(/tmp/ltx2_sync_clean_small_pr.mp4) = 5bc66ef7d4dbc0074195a1d7ed01a4ef`

Updated MP4 assets with a more presentation-friendly prompt:
- Prompt: `At sunrise, a glowing paper lantern boat drifts through a narrow canal between mossy stone walls, soft fog above the water, the camera slowly gliding forward as golden reflections shimmer across the ripples, cinematic, realistic, highly detailed.`
- Negative prompt: `worst quality, blurry, jittery motion, distorted, oversaturated, artifacts`
- Parameters: `256x256`, `17` frames, `6` inference steps, `guidance_scale=4.0`, `frame_rate=24`, `seed=42`, `cfg_parallel_size=2`
- `../ltx2_cfg_parallel_video/preref.mp4`: preref baseline output
- `../ltx2_cfg_parallel_video/refactor.mp4`: refactored LTX2 output
- `../ltx2_cfg_parallel_video/old_vs_new.mp4`: side-by-side comparison video
- `MD5(/tmp/ltx2_preref_pretty.mp4) = 4a04b80d0729c452eb12f55b32209506`
- `MD5(/tmp/ltx2_refactor_pretty.mp4) = 4a04b80d0729c452eb12f55b32209506`
