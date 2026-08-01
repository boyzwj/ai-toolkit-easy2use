def resolve_flowmatch_patch_size(sd) -> int:
    """Return the latent patch size used to calculate dynamic flow shift."""
    if getattr(sd, 'is_flux', False) or 'flex' in getattr(sd, 'arch', ''):
        # Flux uses patch size 1 internally, but its latents are reduced by 2
        # before entering the transformer, so the effective value here is 2.
        return 2

    transformer_config = getattr(getattr(sd, 'unet', None), 'config', None)
    for field_name in ('patch_size', 'patch'):
        patch_size = getattr(transformer_config, field_name, None)
        if patch_size is not None:
            return patch_size

    # Some custom model wrappers expose the effective patch size directly.
    patch_size = getattr(sd, 'patch_size', None)
    if patch_size is not None:
        return patch_size

    return 1
