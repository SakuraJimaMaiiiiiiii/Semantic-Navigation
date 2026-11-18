"""Recoverable SAM3 transport failures, distinct from invalid model output."""


class SAM3TransportError(ConnectionError):
    """A request failed before a usable inference result was received."""

    def __init__(self, message, *, phase=None):
        super().__init__(message)
        self.phase = phase
