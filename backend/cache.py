"""
Minimal in-process TTL cache decorator, standing in for Streamlit's
st.cache_data(ttl=...) outside of a Streamlit runtime. Keyed on function
name + args/kwargs; stdlib only.
"""
import time
import functools


def ttl_cache(ttl_seconds: float = 1800):
    def decorator(fn):
        store = {}

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            now = time.time()
            cached = store.get(key)
            if cached is not None and (now - cached[0]) < ttl_seconds:
                return cached[1]
            result = fn(*args, **kwargs)
            store[key] = (now, result)
            return result

        wrapper.cache_clear = store.clear
        return wrapper
    return decorator
