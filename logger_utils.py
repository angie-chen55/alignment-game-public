def maybe_log(logger, msg, level="info"):
    if logger is None:
        return
    logger_fn = getattr(logger, level)
    logger_fn(msg)
