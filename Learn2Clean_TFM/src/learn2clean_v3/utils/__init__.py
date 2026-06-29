import logging


class LoggingMixin:
    """Lazy per-instance logger."""

    @property
    def log(self) -> logging.Logger:
        if not hasattr(self, "_log") or self._log is None:
            self._log = logging.getLogger(
                f"{self.__class__.__module__}.{self.__class__.__name__}"
            )
        return self._log
