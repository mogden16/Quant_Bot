def atr_stop_target(entry: float, atr: float, stop_mult: float, rr: float):
    stop = entry - stop_mult * atr
    target = entry + rr * (entry - stop)
    return float(stop), float(target)
