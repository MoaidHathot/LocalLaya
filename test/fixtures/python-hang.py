"""A blocked inference behind a real venv launcher, for process-tree timeout verification."""
import json
import os
import sys
import time

print(json.dumps({"ready": True, "model": "vega", "device": "cpu", "maxLen": 73728,
                  "loadMs": 0, "pid": os.getpid()}), flush=True)
for line in sys.stdin:
    message = json.loads(line)
    if message.get("op") == "close":
        break
    time.sleep(3600)
