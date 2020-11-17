import asyncio
import logging
from typing import Coroutine

import click

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


async def _write_hdf_files(host: str, scheme: str, num: int, arm: bool):
    # Local import as we might not have h5py installed and want other commands to work
    from pandablocks.hdf import write_hdf_files

    async with AsyncioClient(host) as client:
        await write_hdf_files(client, scheme=scheme, num=num, arm=arm)


@click.group(invoke_without_command=True)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(
        ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"], case_sensitive=False
    ),
)
@click.version_option()
@click.pass_context
def pandablocks(ctx, log_level: str):
    """PandaBlocks client library command line interface."""

    level = getattr(logging, log_level.upper(), None)
    logging.basicConfig(format="%(levelname)s:%(message)s", level=level)

    # if no command is supplied, print the help message
    if ctx.invoked_subcommand is None:
        click.echo(pandablocks.get_help(ctx))


@pandablocks.command()
@click.option(
    "--num", help="Number of collections to capture", default=1, show_default=True,
)
@click.option(
    "--arm",
    help="Arm PCAP at the start, and after each successful acquisition",
    is_flag=True,
)
@click.argument("host")
@click.argument("scheme")
def hdf(host: str, scheme: str, num: int, arm: bool):
    """
    Write an HDF file for each PCAP acquisition for HOST

    Uses the filename pattern specified by SCHEME, including %d for scan number
    starting from 1
    """
    # Don't use asyncio.run to workaround Python3.7 bug
    # https://bugs.python.org/issue38013
    asyncio_run(_write_hdf_files(host, scheme, num, arm))


@pandablocks.command()
@click.option("--prompt", help="Prompt character", default=PROMPT, show_default=True)
@click.option(
    "--no-readline", help="Disable readline history and completion", is_flag=True,
)
@click.argument("host", type=str)
def control(host: str, prompt: str, no_readline: bool):
    """Open an interactive control console to HOST"""
    interactive_control(host, prompt, not no_readline)
