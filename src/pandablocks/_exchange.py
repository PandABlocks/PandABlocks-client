from collections.abc import Generator
from typing import TypeVar, Union

T = TypeVar("T")


class Exchange:
    """A helper class representing the lines to send to PandA and
    the lines received"""

    def __init__(self, to_send: Union[str, list[str]]):
        if isinstance(to_send, str):
            self.to_send = [to_send]
        else:
            self.to_send = to_send
        self.received: list[str] = []
        self.is_multiline = False

    @property
    def line(self) -> str:
        """Check received is not multiline and return the line"""
        assert not self.is_multiline
        return self.received[0]

    @property
    def multiline(self) -> list[str]:
        """Return the multiline received lines, processed to remove markup"""
        assert self.is_multiline
        # Remove the ! and . markup
        return [line[1:] for line in self.received[:-1]]


Exchanges = Union[Exchange, list[Exchange]]
ExchangeGenerator = Generator[Exchanges, None, T]
