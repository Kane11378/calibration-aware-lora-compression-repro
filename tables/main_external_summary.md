**All eight external transfers on eight reused DeBERTa sources. Counts use equal-rank source means; KL/CE columns count H1-favorable point estimates, and the KL-CI column counts intervals entirely favoring H1.**

| Transfer | Raw KL | KL CI | Raw CE | Cal. CE | Coefficient budget |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Surgeon U | 8/8 | 7/8 | 4/8 | 5/8 | 512/source, labels |
| Surgeon S | 8/8 | 6/8 | 4/8 | 4/8 | 512/source, labels |
| PARA count | 6/8 | 6/8 | 0/8 | 4/8 | no derivatives |
| PARA parameter | 8/8 | 7/8 | 0/8 | 6/8 | no derivatives |
| Spectral abs | 8/8 | 8/8 | 3/8 | 6/8 | 128/M0-k, labels |
| Spectral smooth | 8/8 | 8/8 | 3/8 | 7/8 | 128/M0-k, labels |
| Spectral random | 8/8 | 8/8 | 1/8 | 6/8 | random control |
| Spectral grad | 8/8 | 8/8 | 0/8 | 6/8 | 128/M0-k, labels |

The 64 pairings are related source/configuration comparisons, not independent replications. PARA count favors the external method on QNLI 101/137. Spectral random is a diagnostic; the declared panel shares the 128-example M0-state collection, while the random coefficients ignore its scores.
