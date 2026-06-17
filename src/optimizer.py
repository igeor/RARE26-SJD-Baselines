import torch.optim as optim


def build_optimizer(model, args):
    head_lr = args.lr
    backbone_lr = getattr(args, "backbone_lr", head_lr)
    weight_decay = getattr(args, "weight_decay", 0.0)

    param_groups = [
        {
            "params": [p for p in model.head.parameters() if p.requires_grad],
            "lr": head_lr,
        }
    ]

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    if backbone_params:
        param_groups.insert(
            0,
            {
                "params": backbone_params,
                "lr": backbone_lr,
            },
        )

    return optim.AdamW(param_groups, weight_decay=weight_decay)
