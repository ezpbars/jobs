"""a job for use in examples"""
import asyncio
import math
import os
import time
import aiohttp
from itgs import Itgs
from graceful_death import GracefulDeath
import numpy as np


async def execute(
    itgs: Itgs,
    gd: GracefulDeath,
    sub: str,
    pbar_name: str,
    uid: str,
    duration: int,
    stdev: int,
):
    """an example job which computes a random number between 1 and 1000

    Args:
        itgs (Itgs): the integration to use; provided automatically
        gd (GracefulDeath): the signal tracker; provided automatically
    """
    stdev_step = math.sqrt((stdev * stdev) / duration)
    rng = np.random.default_rng()
    async with Itgs() as itgs:
        async with aiohttp.ClientSession() as session:
            for second in range(0, duration + 1):
                await session.post(
                    url=f'{os.environ["ROOT_BACKEND_URL"]}/api/1/progress_bars/traces/steps/',
                    json={
                        "pbar_name": pbar_name,
                        "trace_uid": uid,
                        "step_name": "calculating",
                        "iteration": second,
                        "iterations": duration,
                        "done": second == duration,
                        "now": time.time(),
                    },
                )
                await asyncio.sleep(
                    max(rng.standard_normal(1)[0] * stdev_step + 1, 0.01)
                )
        redis = await itgs.redis()
        redis.set(f"example:{uid}", bytes(str(rng.integers(1, 1000)), "utf-8"), ex=600)
