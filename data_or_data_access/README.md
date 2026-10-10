# Data and model access

Static verification needs no model or dataset download. Experimental
reproduction uses the following provider resources at immutable revisions:

| Resource | Provider | Revision |
| --- | --- | --- |
| GLUE SST-2/QNLI | https://huggingface.co/datasets/nyu-mll/glue | bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c |
| BoolQ | https://huggingface.co/datasets/google/boolq | 35b264d03638db9f4ce671b711558bf7ff0f80d5 |
| DeBERTa-v3-base | https://huggingface.co/microsoft/deberta-v3-base | 8ccc9b6f36199bec6961081d44eb72fb3f7353f3 |
| RoBERTa-base | https://huggingface.co/FacebookAI/roberta-base | e2da8e2f811d1448a5b465c236feacd80ffbac7b |

The partition records contain benchmark IDs or row positions, partition
hashes, tokenizer settings, aggregate class counts and parquet identities.
Match the pinned revision before interpreting positions. IDs are linkable to
public benchmark rows; source text and individual task labels are omitted.
The original fitting, calibration and evaluation memberships are preserved.
SST-2 official validation is distinct from the development endpoint.

Trained source and compressed adapters, Jacobians, optimizer/RNG checkpoints
and logits are omitted. Hash indexes identify archived assets but do not
provide them. The corresponding author can discuss access to the original
project assets and execution materials. Observe each provider's licensing and
access terms when obtaining upstream resources.
