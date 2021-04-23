import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterator, List

import numpy as np

from pandablocks.asyncio import AsyncioClient
from pandablocks.commands import Put, SetState

STATEFILE = Path(__file__).parent / "qd_spinner.sav"

NUMBERS = """\
 XXX    X   XXXX  XXXX     X  XXXX   XXX  XXXXX  XXX   XXX        |
X   X   X       X     X   XX  X     X         X X   X X   X   X   |
X   X   X      X  XXXX   X X  XXXX  XXXX     X   XXX   XXXX       |
X   X   X    X        X XXXXX     X X   X   X   X   X     X   X   |
 XXX    X   XXXXX XXXX     X  XXXX   XXX   X     XXX   XXX        |
""".splitlines()


def bits(line: int, number: int) -> List[int]:
    return [ord(x) // ord("X") for x in NUMBERS[line][number * 6 : number * 6 + 6]]


def make_seq_table() -> Iterator[str]:
    # Make a sequencer table for the current time
    now = datetime.now()
    out = {}
    for line in range(5):
        out[line] = np.array(
            bits(line, now.hour // 10)
            + bits(line, now.hour % 10)
            + bits(line, 10)[1:-1]
            + bits(line, now.minute // 10)
            + bits(line, now.minute % 10)
            + bits(line, 10)[1:-1]
            + bits(line, now.second // 10)
            + bits(line, now.second % 10)
        )
    # Rows for the numbers
    # trigger on BITA=1, 1 repeat, and our outputs
    words = 0x20001 + np.sum([v << 26 + k for k, v in out.items()], axis=0)
    for word in words:
        yield str(word)  # trig, repeats
        yield "0"  # position
        yield "0"  # time1
        yield "1"  # time2


async def main():
    state = STATEFILE.read_text().splitlines()
    async with AsyncioClient(sys.argv[1]) as client:
        await client.send(SetState(state))
        await client.send(Put("SRGATE1.FORCE_SET", ""))
        seq_num = 0
        while True:
            await client.send(Put(f"SEQ{seq_num+1}.TABLE", list(make_seq_table())))
            await client.send(Put("BITS.A", str(seq_num)))
            seq_num ^= 1
            await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
