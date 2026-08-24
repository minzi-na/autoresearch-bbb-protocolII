# maccs+scage1+mole — architecture log

val: per-seed mean±std  |  subsets (int/ext/nn03/nn05/total): soft voting ensemble single values.

| iter | commit | val_auc | int | ext | nn03 | nn05 | total | note |
|------|--------|---------|-----|-----|------|------|-------|------|
| 1 | 1314ededa0 | 0.8454±0.0029 | 0.9015 | 0.8777 | 0.8799 | 0.8723 | 0.8839 | baseline reproduction |
| 7 | f581594c60 | 0.8456±0.0026 | 0.9018 | 0.8790 | 0.8799 | 0.8741 | 0.8850 | sigmoid skip-gate mix pool |
| 9 | b7fa5ff3fb | 0.8456±0.0027 | 0.9051 | 0.8767 | 0.8669 | 0.8723 | 0.8837 | attention pooling |
| 17 | 4e4b8698b9 | 0.8473±0.0040 | 0.8906 | 0.8766 | 0.8669 | 0.8741 | 0.8811 | EMA weights for ES snapshot |
| 18 | d385f6a489 | 0.8477±0.0033 | 0.8964 | 0.8758 | 0.8604 | 0.8797 | 0.8816 | modality dropout p=0.10 |
| 21 | e679aad281 | 0.8483±0.0034 | 0.8942 | 0.8781 | 0.8669 | 0.8796 | 0.8833 | grad-clip only (1.0) |
| 25 | ba24a1f364 | 0.8484±0.0035 | 0.8916 | 0.8782 | 0.8604 | 0.8770 | 0.8828 | mod_drop p=0.05 |
| 28 | 85e657afda | 0.8484±0.0042 | 0.8924 | 0.8781 | 0.8701 | 0.8775 | 0.8827 | pos_weight=0.6 fixed |
