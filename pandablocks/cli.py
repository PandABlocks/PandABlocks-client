import asyncio
import logging
from argparse import ArgumentParser
from typing import Callable, Coroutine

from pandablocks import __version__
from pandablocks._control import interactive_control
from pandablocks.asyncio import AsyncioClient

# Default prompt
PROMPT = "< "


def asyncio_run(coro: Coroutine):
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(coro)
    finally:
        to_cancel = asyncio.tasks.all_tasks(loop)
        for task in to_cancel:
            task.cancel()
        loop.run_until_complete(
            asyncio.gather(*to_cancel, loop=loop, return_exceptions=True)
        )


async def _write_hdf_files(args):
    # Local import as we might not have h5py installed and want other commands to work
    from pandablocks.hdf import write_hdf_files

    async with AsyncioClient(args.host) as client:
        await write_hdf_files(client, scheme=args.scheme, num=args.num, arm=args.arm)


def hdf(args):
    """Write an HDF file for each PCAP acquisition"""
    # Don't use asyncio.run to workaround Python3.7 bug
    # https://bugs.python.org/issue38013
    asyncio_run(_write_hdf_files(args))


def control(args):
    """Open an interactive control console"""
    interactive_control(args.host, args.prompt, args.readline)


def subparser_with_host(subparsers, func: Callable):
    sub = subparsers.add_parser(func.__name__, help=func.__doc__)
    sub.add_argument("host", type=str, help="IP address for PandA to connect to")
    sub.set_defaults(func=func)
    return sub


def main(args=None):
    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
    parser = ArgumentParser()
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers()
    # hdf subcommand
    sub = subparser_with_host(subparsers, hdf)
    sub.add_argument(
        "scheme",
        type=str,
        help="Filenaming scheme for HDF files, with %%d for scan number starting at 1",
    )
    sub.add_argument(
        "--num", type=int, help="Number of collections to capture, default 1", default=1
    )
    sub.add_argument(
        "--arm",
        action="store_true",
        help="Arm PCAP at the start, and after each successful acquisition",
    )
    # control subcommand
    sub = subparser_with_host(subparsers, control)
    sub.add_argument(
        "--prompt", default=PROMPT, help="Prompt character, default is %r" % PROMPT,
    )
    sub.add_argument(
        "--no-readline",
        action="store_false",
        dest="readline",
        help="Disable readline history and completion",
    )
    # Parse args and run
    args = parser.parse_args(args)
    args.func(args)
