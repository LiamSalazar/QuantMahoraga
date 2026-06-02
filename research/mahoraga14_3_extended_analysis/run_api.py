from __future__ import annotations

import sys
import socket
from pathlib import Path

import uvicorn


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def free_port(start: int = 8000) -> int:
    port = start
    while port < start + 20:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
        port += 1
    raise RuntimeError("No free API port found in range 8000-8019")


if __name__ == "__main__":
    port = free_port(8000)
    print(f"Mahoraga extended analysis API: http://127.0.0.1:{port}")
    uvicorn.run("api.main:app", host="127.0.0.1", port=port, reload=False)
