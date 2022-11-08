"""a very simple job"""
from itgs import Itgs
from graceful_death import GracefulDeath

EXCLUSIVE = False
"""If true, the job will always be run in isolation. Otherwise, other non-exclusive
jobs may be running at the same time. Any cpu-bound job should be marked exclusive,
whereas io-bound jobs should not be marked exclusive.
"""


async def execute(itgs: Itgs, gd: GracefulDeath, *, message: str):
    """an example job execute - this is invoked when 'runners.example' is the name of the job

    Args:
        itgs (Itgs): the integration to use; provided automatically
        gd (GracefulDeath): the signal tracker; provided automatically
        message (str): the only keyword argument for this job; to be printed out
    """
    print(message)
