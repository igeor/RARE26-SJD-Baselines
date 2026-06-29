def get_early_stopping_config(args):
    config = getattr(args, "early_stopping", None) or {}

    if not isinstance(config, dict):
        config = vars(config)

    every = config.get("every", getattr(args, "save_every", args.epochs))
    if every is None:
        every = args.epochs

    return {
        "enabled": bool(config.get("enabled", False)),
        "every": int(every),
        "patience": config.get("patience", None),
        "metric": config.get("metric", "PPV@90% Recall"),
    }
