import inspect
import sys
from datetime import datetime

class CustomLogger:
    LEVELS = {
        "DEBUG": 10,
        "INFO": 20,
        "WARNING": 30,
        "ERROR": 40,
        "CRITICAL": 50
    }

    def __init__(self, level="DEBUG"):
        if level not in self.LEVELS:
            raise ValueError(f"Invalid log level: {level}")
        self.level = self.LEVELS[level]

    def _log(self, level_name, message):
        if self.LEVELS[level_name] < self.level:
            return

        # Get caller info (two frames up: _log -> level method -> user code)
        frame = inspect.stack()[2]
        filename = frame.filename
        lineno = frame.lineno
        funcname = frame.function

        # Format time and message
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = (
            f"[{timestamp}] [{level_name}] "
            f"{filename}:{lineno} in {funcname}() - {message}"
        )
        print(log_message, file=sys.stdout)

    def debug(self, message):
        self._log("DEBUG", message)

    def info(self, message):
        self._log("INFO", message)

    def warning(self, message):
        self._log("WARNING", message)

    def error(self, message):
        self._log("ERROR", message)

    def critical(self, message):
        self._log("CRITICAL", message)