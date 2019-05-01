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
                "fps": 10,
                "depthCameraColor": 3
            },
            "calibration": {
                "checkerboard": {
                    "dimension": [9, 6],
                    "size": 27.5
                },
                "circlesGrid": {
                    "dimension": [
                        11,
                        4
                    ]
                }
            },
            "log": {
                "level": "DEBUG",
                "format": "%(asctime)s %(message)s"
            },
            "openniRedist": "OpenNI directory not set"
        }
        # Save config
        with open("config.json", "w") as f:
            json.dump(_config, f, indent=4, sort_keys=True)

    # Parse
    _config["log"]["level"] = logging.getLevelName(_config["log"]["level"])
    _config["calibration"]["checkerboard"]["dimension"] = tuple(_config["calibration"]["checkerboard"]["dimension"])
    _config["calibration"]["circlesGrid"]["dimension"] = tuple(_config["calibration"]["circlesGrid"]["dimension"])
    _config["openniRedist"] = str(_config["openniRedist"])

    return _config
