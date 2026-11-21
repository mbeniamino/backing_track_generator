def format_time(seconds: float, tenths: bool = True) -> str:
    """Format seconds as m:ss or m:ss.t"""
    seconds = max(seconds, 0.0)
    if tenths:
        total_tenths = int(seconds * 10 + 1e-6)
        m, rem = divmod(total_tenths, 600)
        return f"{m}:{rem // 10:02d}.{rem % 10}"
    m, s = divmod(int(seconds + 1e-6), 60)
    return f"{m}:{s:02d}"
