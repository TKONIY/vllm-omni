# LTX2 CFG-Parallel Parity Artifacts

- `old_vs_new.gif`: left is the preref baseline output; right is the refactored LTX2 output with post-step `torch.cuda.current_stream(...).synchronize()`.
- `diff.gif`: per-frame visual difference between the two videos. It is blank because the outputs are bit-identical.
- `MD5(/tmp/ltx2_preref_small.mp4) = 5bc66ef7d4dbc0074195a1d7ed01a4ef`
- `MD5(/tmp/ltx2_sync_clean_small_pr.mp4) = 5bc66ef7d4dbc0074195a1d7ed01a4ef`
