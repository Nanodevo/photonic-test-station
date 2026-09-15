"""Start the virtual instrument rack and keep it running (Ctrl-C to stop)."""
import time

from pts import virtual_rack

if __name__ == "__main__":
    rack, servers = virtual_rack.start()
    for k, v in virtual_rack.resources().items():
        print(f"{k:12s} {v}")
    print("virtual rack up; Ctrl-C to stop")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
