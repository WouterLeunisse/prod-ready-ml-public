"""Animal shelter outcome prediction package."""

import logging


def setup_logger(*, level: int = logging.INFO) -> None:
    """Set up the logger for the application.

    Args:
        level (int): The logging level to set. Defaults to
            `logging.INFO`.

    """
    logger = logging.getLogger("animal_shelter")

    # Avoid adding duplicate handlers
    if logger.handlers:
        logger.handlers.clear()

    logger.setLevel(level)

    stream_handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


# Initialize logger when module is imported (default to INFO level)
setup_logger(level=logging.INFO)


def set_log_level(level: int | str) -> None:
    """Set the log level of the ``animal_shelter`` logger.

    Parameters
    ----------
    level : int or str
        Logging level, e.g. ``logging.DEBUG`` or ``"debug"``.

    """
    if level in (
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
        logging.ERROR,
        logging.CRITICAL,
    ):
        logging.getLogger("animal_shelter").setLevel(level)
    elif isinstance(level, str):
        level = level.upper()
        if level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            logging.getLogger("animal_shelter").setLevel(getattr(logging, level))


def main() -> None:
    """Print a greeting."""
    print("Hello from animal-shelter!")
