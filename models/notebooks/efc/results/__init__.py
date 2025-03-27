import sys

from loguru import logger

logger.remove()

logger.add(sys.stderr,
           level="INFO",
           format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
           "<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan> "
           "{function}</cyan>:<cyan>{line}</cyan> "
           "- <level>{message}</level> {extra}" if "{extra}" else "<level>{message}</level>"
        )
