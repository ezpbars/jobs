import multiprocessing
import time
import updater
import asyncio
from itgs import Itgs


async def _main():
    multiprocessing.Process(target=updater.listen_forever_sync, daemon=True).start()
    async with Itgs() as itgs:
        slack = await itgs.slack()
        await slack.send_web_error_message("hello from jobs")
    while True:
        print("Hello World")
        time.sleep(5)


def main():
    asyncio.run(_main())


if __name__ == "__main__":
    main()
