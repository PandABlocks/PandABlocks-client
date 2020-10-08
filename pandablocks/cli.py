import asyncio
import logging
from argparse import ArgumentParser
from typing import Callable

from pandablocks import __version__

# Default prompt
PROMPT = "< "


def hdf(args):
    """Write an HDF file for each PCAP acquisition"""
    from pandablocks.hdf import write_hdf_files

    # Don't use asyncio.run to workaround Python3.7 bug
    # https://bugs.python.org/issue38013
    loop = asyncio.get_event_loop()
    loop.run_until_complete(write_hdf_files(args.host, args.scheme, args.num))


def control(args):
    """Open an interactive control console"""
    from pandablocks.control import control

    control(args.host, args.prompt, args.readline)


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
        help="Filenaming scheme for HDF files, with %%d for scan number",
    )
    sub.add_argument(
        "--num", type=int, help="Number of collections to capture, default 1", default=1
    )
    # control subcommand
    sub = subparser_with_host(subparsers, control)
    sub.add_argument(
        "--prompt", default=PROMPT, help="Prompt character, default is a panda face",
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
