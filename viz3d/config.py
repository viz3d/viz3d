import json
import logging

_config = None

def read_config():
    """
    :return: Project specific config
    """

    global _config

    # Check if config was already loaded
    if _config is not None:
        return _config

    with open("config.json") as f:
        # Load config
        config = json.load(f)
        # Parse
        config["log"]["level"] = logging.getLevelName(config["log"]["level"])
        config["calibration"]["checkerboard"]["dimension"] = tuple(config["calibration"]["checkerboard"]["dimension"])
        # Store config
        _config = config
        return _config
