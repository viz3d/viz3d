import json
import logging
import os.path

_config = None


def read_config():
    """
    :return: Project specific config
    """

    global _config

    # Check if config was already loaded
    if _config is not None:
        return _config

    if os.path.isfile("config.json"):
        # Load config
        with open("config.json") as f:
            _config = json.load(f)
    else:
        # Create default config
        _config = {
            "workingDir": ".",
            "cameras": {
                "left": 2,
                "right": 0,
                "fps": 10
            },
            "calibration": {
                "checkerboard": {
                    "dimension": [9, 6],
                    "size": 27.5
                }
            },
            "log": {
                "level": "DEBUG",
                "format": "%(asctime)s %(message)s"
            }
        }
        # Save config
        with open("config.json", "w") as f:
            json.dump(_config, f)

    # Parse
    _config["log"]["level"] = logging.getLevelName(_config["log"]["level"])
    _config["calibration"]["checkerboard"]["dimension"] = tuple(_config["calibration"]["checkerboard"]["dimension"])

    return _config
