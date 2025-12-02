from collections.abc import Generator
from typing import TypeVar

T = TypeVar("T")


class Exchange:
    """A helper class representing the lines to send to PandA and
    the lines received"""

    def __init__(self, to_send: str | list[str]):
        if isinstance(to_send, str):
            self.to_send = [to_send]
        else:
            self.to_send = to_send
        self.received: list[str] = []
        self.is_multiline = False

    def _error_message(self) -> str:
        sent = "\n".join(self.to_send)
        received = "\n".join(self.received)
        return f"{sent!r} -> {received!r}"

    def check_ok(self):
        assert self.received == ["OK"], self._error_message()

    @property
    def line(self) -> str:
        """Check received is not multiline and return the line"""
        assert not self.is_multiline and self.received[0].startswith("OK ="), (
            self._error_message()
        )
        # Remove the OK= header
        return self.received[0][4:]

    @property
    def multiline(self) -> list[str]:
        """Return the multiline received lines, processed to remove markup"""
        assert self.is_multiline, self._error_message()
        # Remove the ! and . markup
        return [line[1:] for line in self.received[:-1]]


Exchanges = Exchange | list[Exchange]
ExchangeGenerator = Generator[Exchanges, None, T]
