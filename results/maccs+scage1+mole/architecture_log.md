# maccs+scage1+mole — architecture log

val: per-seed mean±std  |  subsets (int/ext/nn03/nn05/total): soft voting ensemble single values.

| iter | commit | val_auc | int | ext | nn03 | nn05 | total | note |
|------|--------|---------|-----|-----|------|------|-------|------|
| 1 | ea3efc7c3c | 0.8454±0.0029 | 0.9015 | 0.8777 | 0.8799 | 0.8723 | 0.8839 | baseline reproduction |
| 7 | d420257c88 | 0.8456±0.0026 | 0.9018 | 0.8790 | 0.8799 | 0.8741 | 0.8850 | sigmoid skip-gate mix gated/mean pool |
| 10 | b1bbe9f895 | 0.8456±0.0027 | 0.9044 | 0.8757 | 0.8701 | 0.8724 | 0.8829 | learnable exp-gated residual scale on SGU spatial path |
| 17 | ba612000be | 0.8468±0.0042 | 0.8914 | 0.8774 | 0.8831 | 0.8756 | 0.8820 | EMA(0.999) of weights for val and early-stop snapshot |
| 18 | 7bcb658c33 | 0.8476±0.0037 | 0.8900 | 0.8780 | 0.8636 | 0.8781 | 0.8825 | modality dropout p=0.05 (synthesis with EMA arch) |
| 20 | d61eaa4fa6 | 0.8488±0.0030 | 0.8938 | 0.8782 | 0.8669 | 0.8763 | 0.8830 | multi-head SGU (n_heads=2) synthesis with EMA arch |
| 26 | d56e397571 | 0.8491±0.0027 | 0.8911 | 0.8787 | 0.8701 | 0.8780 | 0.8832 | AdamW + grad-clip(1.0) without scheduler |
| 29 | 7bf5b98346 | 0.8492±0.0029 | 0.8912 | 0.8777 | 0.8701 | 0.8789 | 0.8820 | skip_gate init=1.0 (sigmoid=0.73, gated bias) |
