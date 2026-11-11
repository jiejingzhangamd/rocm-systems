"""Validate the opt-in dispatch strict-window ABI of a built SDK library."""

import argparse
import ctypes
import json


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sdk_library")
    args = parser.parse_args()
    sdk = ctypes.CDLL(args.sdk_library)
    setter = sdk.rocprofiler_experimental_set_dispatch_strict_window
    getter = sdk.rocprofiler_experimental_get_dispatch_strict_window
    contains = sdk.rocprofiler_experimental_dispatch_strict_window_contains
    wait_stats = sdk.rocprofiler_experimental_get_dispatch_strict_bound_wait
    setter.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
    getter.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]
    contains.argtypes = [ctypes.c_uint64]
    contains.restype = ctypes.c_int
    wait_stats.argtypes = [ctypes.POINTER(ctypes.c_uint64)] * 4

    def bounds():
        start, end = ctypes.c_uint64(), ctypes.c_uint64()
        getter(ctypes.byref(start), ctypes.byref(end))
        return start.value, end.value

    def wait_values():
        values = [ctypes.c_uint64() for _ in range(4)]
        wait_stats(*(ctypes.byref(value) for value in values))
        return tuple(value.value for value in values)

    disabled = (2**64 - 1, 0)
    require(bounds() == disabled, "The strict window must be disabled by default")
    require(wait_values() == (0, 0, 0, 100000), "Unexpected default bound-wait observations")
    require(not contains(0) and not contains(2**64 - 1),
            "A disabled window selected a dispatch")
    setter(100, 200)
    require(bounds() == (100, 200), "Strict window readback differs from the set bounds")
    require(wait_values() == (0, 0, 0, 100000), "New window did not reset wait observations")
    for timestamp, expected in ((99, False), (100, True), (150, True),
                                (200, True), (201, False)):
        require(bool(contains(timestamp)) is expected,
                f"Wrong enqueue timestamp selection: {timestamp}")
    setter(200, 100)
    require(bounds() == disabled and not contains(150),
            "An invalid window did not disable strict selection")
    setter(0, 0)
    require(bounds() == disabled, "Zero bounds did not disable strict selection")
    print(json.dumps({"status": "PASS", "default_off": True,
                      "selected_enqueue_bounds": [100, 200],
                      "outside_enqueue_rejected": [99, 201],
                      "invalid_window_disables": True,
                      "bounded_wait_limit_ns": 100000}))


if __name__ == "__main__":
    main()
