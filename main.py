import multiprocessing
import time
from graceful_death import GracefulDeath
import updater
import asyncio
from itgs import Itgs
import importlib
from error_middleware import handle_error
from typing import Any, Dict, Optional, Set, TypedDict

MAX_CONCURRENT = multiprocessing.cpu_count() * 2
"""Maximum number of concurrent non-exclusive jobs"""


class Job(TypedDict):
    name: str
    kwargs: Dict[str, Any]


async def _main(gd: GracefulDeath):
    multiprocessing.Process(target=updater.listen_forever_sync, daemon=True).start()
    async with Itgs() as itgs:
        jobs = await itgs.jobs()

        running: Set[asyncio.Task] = set()
        module_is_exclusive: Dict[str, bool] = dict()
        while not gd.received_term_signal:
            job: Optional[Job] = await jobs.retrieve(timeout=5)
            if job is None:
                running = set(t for t in running if not t.done())
                continue

            if job["name"] not in module_is_exclusive:
                try:
                    mod = importlib.import_module(job["name"])
                    module_is_exclusive[job["name"]] = getattr(mod, "EXCLUSIVE", False)
                except Exception as e:
                    await handle_error(e)
                    continue

            if module_is_exclusive[job["name"]]:
                await asyncio.wait(running, return_when=asyncio.ALL_COMPLETED)
                running.clear()
                await _run_job(itgs, gd, job)
                continue

            while len(running) >= MAX_CONCURRENT:
                _, running = await asyncio.wait(
                    running, return_when=asyncio.FIRST_COMPLETED
                )

            running.add(asyncio.create_task(_run_job(itgs, gd, job)))


async def _run_job(itgs: Itgs, gd: GracefulDeath, job: Job) -> None:
    try:
        mod = importlib.import_module(job["name"])
        started_at = time.perf_counter()
        await mod.execute(itgs, gd, **job["kwargs"])
        print(
            f"finished {job['name']} in {time.perf_counter() - started_at:.3f} seconds"
        )
    except Exception as e:
        await handle_error(e)


def main():
    gd = GracefulDeath()
    asyncio.run(_main(gd))


if __name__ == "__main__":
    main()
