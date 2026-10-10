**Raw rank 32 BoolQ source quality and predicted-class counts. Class labels are false=0 and true=1. The evaluation labels contain 1,237 class 0 and 2,033 class 1 examples for every source.**

| Backbone | Seed | Raw CE | Acc. (%) | Pred.0 | Pred.1 | Quality |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| DeBERTa | 307 | 0.50384 | 85.29 | 1194 | 2076 | nondegenerate |
| DeBERTa | 401 | 0.48916 | 85.54 | 1198 | 2072 | nondegenerate |
| DeBERTa | 503 | 0.41480 | 83.98 | 1181 | 2089 | nondegenerate |
| RoBERTa | 607 | 0.66322 | 62.17 | 0 | 3270 | majority only |
| RoBERTa | 701 | 0.56453 | 79.33 | 1235 | 2035 | nondegenerate |
| RoBERTa | 809 | 0.58780 | 80.61 | 1185 | 2085 | nondegenerate |
