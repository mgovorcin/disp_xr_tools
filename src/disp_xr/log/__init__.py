import os
import json
import logging
from pathlib import Path


def setup_logging(*, logger_name: str = "disp_xr", debug: bool = False, filename: str = None):
    config_file = Path(__file__).parent / "log-config.json"

    with open(config_file) as f_in:
        config = json.load(f_in)

    if logger_name not in config["loggers"]:
        config["loggers"][logger_name] = {"level": "INFO", "handlers": ["stderr"]}

    if debug:
        config["loggers"][logger_name]["level"] = "DEBUG"

    # Ensure "stderr" is always present to print logs to the screen
    if "stderr" not in config["loggers"][logger_name]["handlers"]:
        config["loggers"][logger_name]["handlers"].append("stderr") 

    if filename:
        if "file" not in config["loggers"][logger_name]["handlers"]:
            config["loggers"][logger_name]["handlers"].append("file")
        config["handlers"]["file"]["filename"] = os.fspath(filename) 
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

    if "filename" not in config["handlers"]["file"]:
        config["handlers"].pop("file", None)

    logging.config.dictConfig(config)