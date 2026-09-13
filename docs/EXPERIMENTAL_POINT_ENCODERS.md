# Experimental Point Encoders

The inherited `ptv3_encoder.py` and `sonata_encoder.py` construct per-point MLPs.
They do not implement either Transformer architecture. Historical integration
summaries and launcher examples are retained as migration provenance, not as
validated implementation or performance claims.

DP3 supports the PointNet encoder by default. Selecting `use_ptv3_encoder` or
`use_sonata_encoder` now fails unless
`experimental_allow_encoder_placeholders=true` is explicitly set. This opt-in
is only for reproducing the legacy MLP experiments. Real pretrained checkpoint
paths are rejected instead of silently loading zero matching tensors. Loading a
legacy full policy checkpoint may require the explicit flag; it does not turn
the checkpoint into a pretrained Transformer.

A genuine implementation still needs an external model adapter, its input
preprocessing and feature contract, compatible weights, and separate tests.
