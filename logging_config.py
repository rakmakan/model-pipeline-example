import logging
import os


def setup_logging(level: str | None = None) -> None:
    level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    numeric = getattr(logging, level, None)
    if not isinstance(numeric, int):
        raise ValueError(f"Invalid LOG_LEVEL: {level!r}")
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
