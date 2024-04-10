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
PIPELINE_STAGE = os.getenv("PIPELINE_STAGE", None)

logging.basicConfig(
    filename=Path(log) / "raw.log",
    level=logging.DEBUG,
    filemode="w",
    format="%(asctime)s - %(message)s",
)


from main import consume_dir

# match PIPELINE_STAGE:
#     case "RAW":
#         from raw import consume_dir
#     case "INFER":
#         from infer import consume_dir
#     case "SEGMENTATION":
#         from segmentation import consume_dir
#     case _:
#         raise ValueError(f"Invalid pipeline stage {PIPELINE_STAGE}")

# from infer import consume_dir

# pipeline_functions = {
#     "RAW": consume_dir,
#     "INFER": consume_dir,
#     "SEGMENTATION": consume_dir,
# }
# pipeline_function = pipeline_functions[PIPELINE_STAGE]


class MyHandler(FileSystemEventHandler):
    def on_modified(self, event):
        logging.info(f"Event type: {event.event_type}  path : {event.src_path}")
        consume_dir(Path(input_dir), Path(output_dir))


if __name__ == "__main__":
    event_handler = MyHandler()
    observer = Observer()
    observer.schedule(event_handler, path=input_dir, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
