from pathlib import Path
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os
import subprocess

import logging


log = os.getenv("LOG_DIR", "/data/log")
input_dir = os.getenv("INPUT_DIR", "/data/sonar")
output_dir = os.getenv("OUTPUT_DIR", "/data/processed")

# Guard against identical input and output paths
if Path(input_dir).resolve() == Path(output_dir).resolve():
    logging.error("INPUT_DIR and OUTPUT_DIR must be different to avoid infinite processing loops.")
    raise ValueError("INPUT_DIR and OUTPUT_DIR are identical")

logging.basicConfig(
    filename=Path(log) / "raw.log",
    level=logging.DEBUG,
    filemode="w",
    format="%(asctime)s - %(message)s",
)


from main import consume_dir

class MyHandler(FileSystemEventHandler):
    def on_modified(self, event):
        logging.info(f"Event type: {event.event_type}  path : {event.src_path}")
        if Path(input_dir).resolve() != Path(output_dir).resolve():
            consume_dir(Path(input_dir), Path(output_dir))

    def on_any_event(self, event):
        # React to *created* and *modified* for files only
        if event.event_type in ("created", "modified"):
            if Path(input_dir).resolve() != Path(output_dir).resolve():
                consume_dir(Path(input_dir), Path(output_dir))


if __name__ == "__main__":
    consume_dir(Path(input_dir), Path(output_dir))
    observer = Observer()
    observer.schedule(MyHandler(), path=input_dir, recursive=True)
    observer.start()

    try:
        while True:
            # Periodic catch‑all pass every 10s
            if Path(input_dir).resolve() != Path(output_dir).resolve():
                consume_dir(Path(input_dir), Path(output_dir))
            time.sleep(10)  # 10‑second interval
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
