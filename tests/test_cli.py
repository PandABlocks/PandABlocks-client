import subprocess
import sys

from pandablocks import __version__


def test_cli_version():
    cmd = [sys.executable, "-m", "pandablocks", "--version"]
    assert subprocess.check_output(cmd).decode().strip() == __version__
