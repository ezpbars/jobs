import multiprocessing
import time
from graceful_death import GracefulDeath
import updater
import asyncio
from itgs import Itgs
import importlib
from error_middleware import handle_error


async def _main(gd: GracefulDeath):
    multiprocessing.Process(target=updater.listen_forever_sync, daemon=True).start()
    async with Itgs() as itgs:
        jobs = await itgs.jobs()
        while not gd.received_term_signal:
            job = await jobs.retrieve(timeout=5)
            if job is None:
                continue
            try:
                mod = importlib.import_module(job["name"])
                await mod.execute(itgs, gd, **job["kwargs"])
            except Exception as e:
                await handle_error(e)
                continue


def main():
    gd = GracefulDeath()
    asyncio.run(_main(gd))


if __name__ == "__main__":
    main()
