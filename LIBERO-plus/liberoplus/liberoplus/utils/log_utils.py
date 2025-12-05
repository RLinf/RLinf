import logging
from termcolor import colored


class LiberoColorFormatter(logging.Formatter):
    """This color format is for logging user's project wise information"""

    format_str = "[Project %(levelname)s] "
    debug_message_str = "%(message)s (%(filename)s:%(lineno)d)"
    message_str = "%(message)s"
    FORMATS = {
        logging.DEBUG: format_str + debug_message_str,
        logging.INFO: message_str,
        logging.WARNING: colored(format_str, "yellow", attrs=["bold"]) + message_str,
        logging.ERROR: colored(format_str, "red", attrs=["bold"]) + message_str,
        logging.CRITICAL: colored(format_str, "red", attrs=["bold", "reverse"])
        + message_str,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class LiberoDefaultLogger:
    def __init__(self, logger_config_path, project_name="libero"):
        config = YamlConfig(logger_config_path).as_easydict()
        config["loggers"][project_name] = config["loggers"]["project"]
        os.makedirs("logs", exist_ok=True)
        logging.config.dictConfig(config)


ProjectDefaultLogger(logger_config_path, project_name)


def get_project_logger(project_name="libero", logger_config_path=None):
    """This function returns a logger that follows the deoxys convention"""
    logger = logging.getLogger(project_name)
    return logger
