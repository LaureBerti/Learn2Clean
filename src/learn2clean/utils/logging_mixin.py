import logging


class LoggingMixin:
    """
    Mixin that provides a logger instance named after the inheriting class.

    The logger is initialized lazily (on first access) and cached per instance.
    """

    _log: logging.Logger | None = None

    @property
    def log(self) -> logging.Logger:
        """
        Returns the logger instance for this object.
        Initializes the logger on first access.
        """
        if self._log is None:
            name = f"{self.__class__.__module__}.{self.__class__.__name__}"
            self._log = logging.getLogger(name)
        return self._log
