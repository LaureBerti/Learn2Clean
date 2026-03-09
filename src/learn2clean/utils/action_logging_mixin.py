from .logging_mixin import LoggingMixin


class ActionLoggingMixin(LoggingMixin):
    """
    Mixin to standardize log message formatting for DataFrameAction subclasses.
    It assumes the inheriting class has a 'name' attribute.
    """

    def _log_prefix(self) -> str:
        """
        Constructs the standard log prefix: [ClassName].
        """
        class_name = self.__class__.__name__
        return f"[{class_name}] "

    def log_debug(self, message: str) -> None:
        self.log.debug(f"{self._log_prefix()}{message}")

    def log_info(self, message: str) -> None:
        self.log.info(f"{self._log_prefix()}{message}")

    def log_warning(self, message: str) -> None:
        self.log.warning(f"{self._log_prefix()}{message}")

    def log_error(self, message: str) -> None:
        self.log.error(f"{self._log_prefix()}{message}")

    def log_exception(self, message: str) -> None:
        self.log.exception(f"{self._log_prefix()}{message}")
