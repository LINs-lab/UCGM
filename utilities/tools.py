import os
import ast
import yaml
import logging
import argparse
import torch.distributed as dist


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def create_logger(logging_dir, logger_name, use_color=True):
    """
    Create a logger that writes to a log file and stdout.

    Args:
        logging_dir (str): Directory where the log file will be saved.
        logger_name (str): Name of the logger.
        use_color (bool): Whether to use ANSI colors in the console output.

    Returns:
        logging.Logger: Configured logger.
    """
    # Ensure distributed environment is initialized
    if not dist.is_initialized():
        raise RuntimeError("Distributed process group is not initialized.")

    rank = dist.get_rank()
    logger = logging.getLogger(logger_name)

    # Prevent adding handlers multiple times
    if not logger.handlers:
        if rank == 0:  # Main process
            logger.setLevel(logging.INFO)

            # Ensure log directory exists
            os.makedirs(logging_dir, exist_ok=True)
            log_file = os.path.join(logging_dir, logger_name + "_log.txt")

            # Create file handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_formatter)

            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            if use_color:
                console_formatter = ColoredFormatter(
                    "%(levelname)s: %(message)s",
                )
            else:
                console_formatter = logging.Formatter(
                    "%(levelname)s: %(message)s",
                )
            console_handler.setFormatter(console_formatter)

            # Add handlers to logger
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

            # Prevent log messages from propagating to the root logger
            logger.propagate = False
        else:  # Non-main process
            logger.setLevel(logging.WARNING)  # Set to higher level to avoid output
            logger.addHandler(logging.NullHandler())

    return logger


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds ANSI colors to log levels.
    """

    COLOR_MAP = {
        "DEBUG": "\033[37m",  # White
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Purple
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLOR_MAP.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def parse_list(option_str):
    try:
        option_str = option_str.strip()
        if option_str.startswith("[") and option_str.endswith("]"):
            elements = option_str[1:-1].split(",")
            processed_elements = []
            for element in elements:
                element = element.strip()
                if element.replace(".", "", 1).isdigit():
                    processed_elements.append(element)
                else:
                    processed_elements.append(f'"{element}"')
            option_str = "[" + ", ".join(processed_elements) + "]"

        result = ast.literal_eval(option_str)
        if not isinstance(result, list):
            raise ValueError
        return result
    except:
        raise argparse.ArgumentTypeError("Invalid list format")
