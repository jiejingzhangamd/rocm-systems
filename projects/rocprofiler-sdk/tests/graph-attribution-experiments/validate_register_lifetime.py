"""Validate a native rocprofv3 JSON collected by register_lifetime_probe.py."""
import json
import sys
from pathlib import Path

trace = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["rocprofiler-sdk-tool"][0]
assert trace["metadata"]["fini_time"] > trace["metadata"]["init_time"]
records = trace["buffer_records"]["hip_api"]
names = trace["strings"]["buffer_records"]
registrations = [
    record
    for record in records
    if names[record["kind"]]["operations"][record["operation"]] == "__hipRegisterFunction"
]
assert registrations
for record in registrations:
    args = {arg["name"]: arg["value"] for arg in record["args"]}
    assert int(args["deviceFunction"], 16) >= 0
    assert int(args["deviceName"], 16) >= 0
print("PASS: completed native JSON with pointer-only registration arguments")
