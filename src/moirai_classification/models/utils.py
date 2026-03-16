def unfreeze_only_moirai_mask(encoder):
    for param in encoder.parameters():
        param.requires_grad = False
    for name, param in encoder.named_parameters():
        if "mask" in name.lower():
            param.requires_grad = True
