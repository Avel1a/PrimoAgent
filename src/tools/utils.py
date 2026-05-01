from datetime import datetime
from typing import Any, Optional
from dataclasses import dataclass

@dataclass
class ToolResult:
    """Simple result format for all tools."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now() 

_yf_session = None

def get_yf_session():
    global _yf_session
    if _yf_session is not None:
        return _yf_session
    from requests_cache import CachedSession
    from requests_ratelimiter import LimiterAdapter, InMemoryBucket
    from pyrate_limiter import Limiter, Rate, Duration
    from datetime import timedelta
    from pathlib import Path
    cache_dir = Path(__file__).resolve().parents[2] / "output" / "yfcache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_name = str(cache_dir / "http_cache")
    s = CachedSession(cache_name=cache_name, expire_after=timedelta(days=1))
    adapter = LimiterAdapter(limiter=Limiter(Rate(2, Duration.SECOND)), bucket_class=InMemoryBucket)
    s.mount("https://", adapter)
    s.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    _yf_session = s
    return _yf_session

_yf_limiter = None

def ratelimit_yf():
    global _yf_limiter
    if _yf_limiter is None:
        from pyrate_limiter import Limiter, Rate, Duration
        _yf_limiter = Limiter(Rate(2, Duration.SECOND))
    _yf_limiter.try_acquire("yfinance", blocking=True)
