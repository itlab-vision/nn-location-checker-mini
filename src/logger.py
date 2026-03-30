import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


def configure_logger(log_folder: Path, filename: str | None) -> None:
    if not log_folder.exists():
        log_folder.mkdir(parents=True)

    file_handler = TimedRotatingFileHandler(
        log_folder.joinpath("latest.log" if filename is None else filename),
        "midnight",
        encoding="utf-8",
    )

    console_handler = logging.StreamHandler()
    format_template = "%(asctime)s %(levelname)s:%(message)s"
    date_template = "%d/%m/%Y %H:%M:%S"
    logging.basicConfig(
        format=format_template,
        datefmt=date_template,
        handlers=[console_handler, file_handler],
    )
