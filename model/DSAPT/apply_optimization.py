"""Isolated-worker execution variants; no saved parameter schema changes.

Install once per process after model construction. Pool, attention and block
patches target this encoder. Geometry metadata uses process-wide Point wrappers
scoped to this encoder's forward; integer encoding changes the process-wide
Point encode binding. Do not mix patched and unpatched encoders in one worker.
For a complete rollback (including attention wrappers), reconstruct the model
in a fresh process. Checkpoints remain loadable by the unmodified implementation.
"""


def apply(encoder, variant):
    if variant in ("baseline", "baseline_repeat"):
        return []
    valid = {"metadata", "pool", "pool_serialization", "block",
             "combined", "combined_serialization", "geometry", "fused_serialization", "all"}
    if variant not in valid:
        raise ValueError(f"Unknown variant: {variant}")
    changed = []
    if variant.startswith("pool") or variant.startswith("combined") or variant == "all":
        from .optimized_pool import patch
        changed.extend(patch(encoder, reuse_serialization=variant.endswith("serialization") or variant == "all"))
    extra_attention = ()
    if variant == "metadata" or variant.startswith("combined") or variant == "all":
        from . import optimized_attention_metadata as metadata
        changed.append(f"attention_metadata:{metadata.patch(encoder)}")
        extra_attention = (metadata._local_forward, metadata._global_forward)
    if variant == "block" or variant.startswith("combined") or variant == "all":
        from .optimized_block import patch
        handle = patch(encoder, attention_forwards=extra_attention)
        # Keep the restore handle on the instance, outside Module registration.
        encoder._exact_block_patch_handle = handle
        changed.append(f"block_dispatch:{handle.block_count}")
    if variant in ("geometry", "all"):
        from . import optimized_geometry_metadata as geometry
        encoder._exact_geometry_patch_handle = geometry.patch(encoder)
        changed.append("batch1_geometry_metadata")
    if variant in ("fused_serialization", "all"):
        from . import optimized_serialization_kernel as serialization
        serialization.patch(encoder)
        changed.append("compiled_integer_serialization")
    return changed
