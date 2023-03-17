import asyncio
import io
import logging
import pathlib
from typing import Awaitable, List

import click
from click.exceptions import ClickException

from pandablocks._control import interactive_control
from pandablocks.asyncio import AsyncioClient
from pandablocks.commands import GetState, SetState, T

# Default prompt
PROMPT = "< "
TUTORIAL = pathlib.Path(__file__).parent / "saves" / "tutorial.sav"


def asyncio_run(coro: Awaitable[T]) -> T:
    loop = asyncio.get_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        to_cancel = asyncio.tasks.all_tasks(loop)
        for task in to_cancel:
            task.cancel()
        loop.run_until_complete(asyncio.gather(*to_cancel, return_exceptions=True))


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
def cli(ctx, log_level: str):
    """PandaBlocks client library command line interface."""

    level = getattr(logging, log_level.upper(), None)
    logging.basicConfig(format="%(levelname)s:%(message)s", level=level)

    # if no command is supplied, print the help message
    if ctx.invoked_subcommand is None:
        click.echo(cli.get_help(ctx))


@cli.command()
@click.option("--prompt", help="Prompt character", default=PROMPT, show_default=True)
@click.option(
    "--no-readline",
    help="Disable readline history and completion",
    is_flag=True,
)
@click.argument("host", type=str)
def control(host: str, prompt: str, no_readline: bool):
    """Open an interactive control console to HOST"""
    interactive_control(host, prompt, not no_readline)


@cli.command()
@click.argument("host")
@click.argument("outfile", type=click.File("w"))
def save(host: str, outfile: io.TextIOWrapper):
    """
    Save the current blocks configuration of HOST to OUTFILE
    """

    async def _save(host: str) -> List[str]:
        async with AsyncioClient(host) as client:
            return await client.send(GetState())

    state = asyncio_run(_save(host))
    outfile.write("\n".join(state) + "\n")


@cli.command()
@click.argument("host")
@click.argument("infile", type=click.File("r"), required=False)
@click.option("--tutorial", help="load the tutorial settings", is_flag=True)
def load(host: str, infile: io.TextIOWrapper, tutorial: bool):
    """
    Load a blocks configuration into HOST using the commands in INFILE
    """
    if tutorial:
        with TUTORIAL.open("r") as stream:
            state = stream.read().splitlines()
    elif infile is None:
        raise ClickException("INFILE not specified")
    else:
        state = infile.read().splitlines()

    async def _load(host: str, state: List[str]):
        async with AsyncioClient(host) as client:
            await client.send(SetState(state))

    asyncio_run(_load(host, state))


try:
    # Local import as we might not have h5py installed and want other commands
    # to work.
    from pandablocks.hdf import write_hdf_files

    @cli.command()
    @click.option(
        "--num",
        help="Number of collections to capture",
        default=1,
        show_default=True,
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

        async def _write_hdf_files(host: str, scheme: str, num: int, arm: bool):
            def file_name_generator(scheme: str):
                """Yield incrementally numbered file names based on provided scheme"""
                counter = 1
                while True:
                    yield scheme % counter
                    counter += 1

            async with AsyncioClient(host) as client:
                await write_hdf_files(
                    client, file_names=file_name_generator(scheme), num=num, arm=arm
                )

        # Don't use asyncio.run to workaround Python3.7 bug
        # https://bugs.python.org/issue38013
        asyncio_run(_write_hdf_files(host, scheme, num, arm))

except ImportError:

    @cli.command(hidden=True)
    def hdf():
        click.echo("ERROR: hdf subcommand unavailable - install 'hdf5' extras.")
