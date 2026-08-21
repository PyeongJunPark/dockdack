"""Domain exceptions for broker integrations."""


class BrokerError(RuntimeError):
    """Base error for all DockDack broker failures."""


class ConfigurationError(BrokerError):
    """Raised when credentials or environment settings are invalid."""


class BrokerAPIError(BrokerError):
    """Raised when Kiwoom returns an HTTP or business-level error."""

    def __init__(
        self,
        message: str,
        *,
        return_code: int | str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.return_code = return_code
        self.status_code = status_code


class LiveOrderConfirmationRequired(BrokerError):
    """Raised when the live-order safety gates have not been satisfied."""


class OptionalDependencyError(BrokerError):
    """Raised when an optional feature dependency is missing."""
