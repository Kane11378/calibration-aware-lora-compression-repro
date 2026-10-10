**Evidence roles. Calibration inputs are train-derived and separate from source fitting. The external panel reuses eight existing DeBERTa sources.**

| Task/backbone | Seeds | Eval. N | Cal. N | Role |
| :-- | :-- | :-- | :-- | :-- |
| SST-2/DeBERTa | 61/73 | 2,048 | 512 | development; train-derived tuning |
| QNLI/DeBERTa | 101/137/211 | 5,463 | 512 | method development; public validation |
| BoolQ/DeBERTa | 307/401/503 | 3,270 | 512 | new-task confirmation; public validation |
| BoolQ/RoBERTa | 607/701/809 | 3,270 | 512 | second-backbone replication; same raw IDs |
