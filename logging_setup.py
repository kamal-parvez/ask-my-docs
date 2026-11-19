"""Logging to a rotating file (logs/chatbot.log)"""
import logging
from logging.handlers import RotatingFileHandler
import config


def setup_logging():
    """Configure the "chatbot" logger"""
    logger = logging.getLogger("chatbot")
    if logger.handlers:
        return
    level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False
    try:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(config.LOG_DIR / "chatbot.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    except OSError:
        handler = logging.NullHandler()  # logging must never stop the app
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
