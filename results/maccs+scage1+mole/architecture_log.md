# maccs+scage1+mole — architecture log

val: per-seed mean±std  |  subsets (int/ext/nn03/nn05/total): soft voting ensemble single values.

| iter | commit | val_auc | int | ext | nn03 | nn05 | total | note |
|------|--------|---------|-----|-----|------|------|-------|------|
| 1 | 10e7b427ca | 0.8454±0.0029 | 0.9015 | 0.8777 | 0.8799 | 0.8723 | 0.8839 | baseline reproduction |
| 7 | 3768306039 | 0.8456±0.0026 | 0.9018 | 0.8790 | 0.8799 | 0.8741 | 0.8850 | skip-gate gated/mean pool |
| 10 | 8b00da3e88 | 0.8456±0.0027 | 0.9044 | 0.8757 | 0.8701 | 0.8724 | 0.8829 | SGU gate_scale |
| 17 | c85402df17 | 0.8468±0.0042 | 0.8914 | 0.8774 | 0.8831 | 0.8756 | 0.8820 | EMA weights decay=0.999 |
| 20 | 80a1a27385 | 0.8476±0.0039 | 0.8906 | 0.8772 | 0.8896 | 0.8750 | 0.8818 | AdamW + grad-clip(1.0) no scheduler |
| 22 | 750e794b82 | 0.8485±0.0035 | 0.8921 | 0.8786 | 0.8701 | 0.8772 | 0.8833 | modality dropout p=0.05 |
| 24 | f56ea03c41 | 0.8490±0.0036 | 0.8930 | 0.8795 | 0.8799 | 0.8789 | 0.8838 | stochastic depth p=0.025 |
| 33 | ceb1a7b0cb | 0.8494±0.0041 | 0.8900 | 0.8788 | 0.8636 | 0.8787 | 0.8831 | modality dropout p=0.075 |
| 50 | 62da7ecb80 | 0.8494±0.0041 | 0.8900 | 0.8788 | 0.8636 | 0.8787 | 0.8831 | AdamW no-decay on bias/norm |
| 58 | 3c344e16a7 | 0.8496±0.0038 | 0.8924 | 0.8791 | 0.8734 | 0.8783 | 0.8834 | triple-pool gated+mean+max |
| 60 | e8411380cd | 0.8496±0.0039 | 0.8924 | 0.8786 | 0.8701 | 0.8785 | 0.8829 | quad-pool gated+mean+max+attention |
| 86 | 6df75b2177 | 0.8503±0.0039 | 0.8923 | 0.8786 | 0.8734 | 0.8753 | 0.8829 | final LayerNorm before pool on quad-pool stack |
| 93 | 2b0f163315 | 0.8503±0.0038 | 0.8924 | 0.8781 | 0.8701 | 0.8751 | 0.8827 | per-pool functional LN before weighted sum |
| 94 | ac4cf46ec9 | 0.8503±0.0039 | 0.8930 | 0.8780 | 0.8701 | 0.8767 | 0.8825 | per-pool LN with learnable affine |
| 100 | ec96cf76b3 | 0.8503±0.0038 | 0.8904 | 0.8790 | 0.8669 | 0.8763 | 0.8830 | multi-head attention pool n_heads=2 |
| 107 | 5087f4403a | 0.8507±0.0034 | 0.8967 | 0.8779 | 0.8604 | 0.8720 | 0.8831 | attn-pool head-mixing linear identity init |
| 109 | eca523955f | 0.8508±0.0035 | 0.8967 | 0.8780 | 0.8636 | 0.8721 | 0.8830 | attn-pool head-mixing linear after LN |
| 132 | 40b8a8a002 | 0.8509±0.0022 | 0.9006 | 0.8790 | 0.8701 | 0.8728 | 0.8847 | 2-layer MLP head |
| 136 | 5e9b2cf885 | 0.8511±0.0024 | 0.9001 | 0.8781 | 0.8604 | 0.8722 | 0.8840 | SiLU activation in MLP head |
| 139 | d83a59b5af | 0.8516±0.0040 | 0.8964 | 0.8779 | 0.8766 | 0.8751 | 0.8836 | wider MLP head 2x d_model |
| 141 | ff7a13f333 | 0.8519±0.0031 | 0.8942 | 0.8784 | 0.8734 | 0.8764 | 0.8831 | 2x MLP head with dropout 0.05 |
| 147 | a99ce806cf | 0.8523±0.0033 | 0.8938 | 0.8806 | 0.8766 | 0.8779 | 0.8848 | mod_drop_p 0.1 |
| 151 | cdfbdeb4ea | 0.8523±0.0039 | 0.8948 | 0.8786 | 0.8701 | 0.8753 | 0.8837 | MLP head dropout before first Linear |
| 152 | 85fec12fc1 | 0.8525±0.0036 | 0.8945 | 0.8787 | 0.8669 | 0.8756 | 0.8836 | input-side MLP head dropout 0.1 |
| 156 | 821652c8dc | 0.8525±0.0036 | 0.8959 | 0.8784 | 0.8701 | 0.8745 | 0.8836 | pool_logits biased toward attn_pool |
| 168 | 0a85eb79fc | 0.8526±0.0036 | 0.8953 | 0.8791 | 0.8766 | 0.8763 | 0.8840 | input-side dropout 0.08 |
| 181 | 64222c5564 | 0.8526±0.0036 | 0.8948 | 0.8792 | 0.8799 | 0.8762 | 0.8843 | pool_logits init [0,0,0,0.7] |
| 196 | f90c886501 | 0.8526±0.0038 | 0.8960 | 0.8792 | 0.8701 | 0.8762 | 0.8844 | n_attn_heads=1 |
| 198 | 2e8cbdc9f5 | 0.8527±0.0036 | 0.8940 | 0.8791 | 0.8701 | 0.8755 | 0.8840 | pool_logits [0,0,0,0.6] H=1 |
