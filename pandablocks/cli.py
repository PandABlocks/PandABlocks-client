import asyncio
import logging
from argparse import ArgumentParser

from pandablocks import __version__


def hdf(args):
    from .hdf import write_hdf_files

    asyncio.run(write_hdf_files(args.host, args.scheme, args.num))


def main(args=None):
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser()
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers()
    # Add a command for writing HDF files
    sub = subparsers.add_parser(
        "hdf", help="Write an HDF file for each PCAP acquisition"
    )
    sub.add_argument("host", type=str, help="IP address for PandA to connect to")
    sub.add_argument(
        "scheme",
        type=str,
        help="Filenaming scheme for HDF files, with %%d for scan number",
    )
    sub.add_argument(
        "--num", type=int, help="Number of collections to capture", default=1
    )
    sub.set_defaults(func=hdf)
    # Parse args
    args = parser.parse_args(args)
    args.func(args)
