"""this job should be called once a trace is completed; it will bootstrap the
progress bar if necessary, and then will sample the trace according to the
progress bar settings"""
import time
from typing import List, Optional, Tuple
from itgs import Itgs
from graceful_death import GracefulDeath
from redis.asyncio import Redis
from dataclasses import dataclass
import secrets
import random
import asyncio


@dataclass
class TraceInfo:
    created_at: float
    last_updated_at: float
    current_step: int
    done: bool


@dataclass
class TraceStepInfo:
    step_name: str
    iteration: Optional[int]
    iterations: Optional[int]
    started_at: float
    finished_at: Optional[float]


@dataclass
class BarInfo:
    version: int


@dataclass
class StepInfo:
    name: str
    iterated: bool


async def execute(
    itgs: Itgs, gd: GracefulDeath, *, user_sub: str, pbar_name: str, trace_uid: str
):
    """potentially bootstraps the progress bar, and then samples the trace according
    to the progress bar sampling technique

    Args:
        itgs (Itgs): the integration to use; provided automatically
        gd (GracefulDeath): the signal tracker; provided automatically
        user_sub (str): the user sub
        pbar_name (str): the name of the progress bar
        trace_uid (str): the uid of the completed trace
    """
    trace_info = await get_info_from_redis(itgs, user_sub, pbar_name, trace_uid)
    if not trace_info:
        return

    async def bounce():
        jobs = await itgs.jobs()
        await jobs.enqueue(
            "runners.handle_completed_trace",
            user_sub=user_sub,
            pbar_name=pbar_name,
            trace_uid=trace_uid,
        )

    if gd.received_term_signal:
        return await bounce()
    pbar_info = await get_bar_info(itgs, user_sub, pbar_name)
    if gd.received_term_signal:
        return await bounce()
    if not pbar_info:
        await asyncio.wait(
            [
                use_trace_to_initialize_pbar(itgs, user_sub, pbar_name, trace_info),
                purge_info_from_redis(itgs, user_sub, pbar_name, trace_uid),
            ]
        )
        return

    steps_info = await get_steps_info(itgs, user_sub, pbar_name, pbar_info.version)
    if gd.received_term_signal:
        return await bounce()
    is_compatible = len(steps_info) == len(trace_info[1]) and all(
        actual.step_name == expected.name
        and (actual.iterations is not None) == expected.iterated
        for actual, expected in zip(trace_info[1], steps_info)
    )
    if not is_compatible:
        await asyncio.wait(
            [
                use_trace_to_increment_pbar_version(
                    itgs, user_sub, pbar_name, pbar_info.version, trace_info, steps_info
                ),
                purge_info_from_redis(itgs, user_sub, pbar_name, trace_uid),
            ]
        )
        return
    await asyncio.wait(
        [
            sample_trace(
                itgs, user_sub, pbar_name, pbar_info.version, trace_info, steps_info
            ),
            purge_info_from_redis(itgs, user_sub, pbar_name, trace_uid),
            update_tcount(
                itgs,
                user_sub,
                pbar_name,
                pbar_info.version,
                trace_uid,
                trace_info[0].created_at,
            ),
        ]
    )


def redis_trace_key(user_sub: str, pbar_name: str, trace_uid: str) -> str:
    return f"trace:{user_sub}:{pbar_name}:{trace_uid}"


async def get_info_from_redis(
    itgs: Itgs, user_sub: str, pbar_name: str, trace_uid: str
) -> Optional[Tuple[TraceInfo, List[TraceStepInfo]]]:
    """gets the trace info from redis

    Args:
        itgs (Itgs): the integration to use
        user_sub (str): the user sub
        pbar_name (str): the name of the progress bar
        trace_uid (str): the uid of the trace

    Returns:
        Optional[Tuple[TraceInfo, List[TraceStepInfo]]]: the trace info and the steps
    """
    redis = await itgs.redis()
    return await redis.transaction(
        (lambda pipe: get_info_from_redis_raw(pipe, user_sub, pbar_name, trace_uid)),
        redis_trace_key(user_sub, pbar_name, trace_uid),
        value_from_callable=True,
    )


async def get_info_from_redis_raw(
    redis: Redis, user_sub: str, pbar_name: str, trace_uid: str
) -> Optional[Tuple[TraceInfo, List[TraceStepInfo]]]:
    trace_info_raw = await redis.hmget(
        redis_trace_key(user_sub, pbar_name, trace_uid),
        "created_at",
        "last_updated_at",
        "current_step",
        "done",
    )
    if not trace_info_raw[0]:
        return None
    trace_info = TraceInfo(
        created_at=float(trace_info_raw[0]),
        last_updated_at=float(trace_info_raw[1]),
        current_step=int(trace_info_raw[2]),
        done=bool(int(trace_info_raw[3])),
    )
    steps_info: List[TraceStepInfo] = []
    for step in range(1, trace_info.current_step + 1):
        step_info_raw = await redis.hmget(
            f"trace:{user_sub}:{pbar_name}:{trace_uid}:step:{step}",
            "step_name",
            "iteration",
            "iterations",
            "started_at",
            "finished_at",
        )
        step_info = TraceStepInfo(
            step_name=step_info_raw[0],
            iteration=int(step_info_raw[1]),
            iterations=int(step_info_raw[2]),
            started_at=float(step_info_raw[3]),
            finished_at=float(step_info_raw[4]) if step_info_raw[4] else None,
        )
        if step_info.iterations == 0:
            step_info.iterations = None
            step_info.iteration = None
        steps_info.append(step_info)

    return trace_info, steps_info


async def purge_info_from_redis(
    itgs: Itgs, user_sub: str, pbar_name: str, trace_uid: str
) -> None:
    """purges the trace info from redis

    Args:
        itgs (Itgs): the integration to use
        user_sub (str): the user sub
        pbar_name (str): the name of the progress bar
        trace_uid (str): the uid of the trace
    """
    redis = await itgs.redis()
    current_step_raw = await redis.hget(
        redis_trace_key(user_sub, pbar_name, trace_uid), "current_step"
    )
    if current_step_raw is None:
        return
    current_step = int(current_step_raw)
    async with redis.pipeline() as pipe:
        pipe.multi()
        await pipe.delete(redis_trace_key(user_sub, pbar_name, trace_uid))
        for step in range(1, current_step + 1):
            await pipe.delete(f"trace:{user_sub}:{pbar_name}:{trace_uid}:step:{step}")
        await pipe.execute()


async def get_bar_info(itgs: Itgs, user_sub: str, pbar_name: str) -> Optional[BarInfo]:
    """gets the progress bar info from the database

    Args:
        itgs (Itgs): the integration to use
        user_sub (str): the user sub
        pbar_name (str): the name of the progress bar

    Returns:
        Optional[BarInfo]: the progress bar info, or None if it doesn't exist
    """
    conn = await itgs.conn()
    cursor = conn.cursor("strong")
    response = await cursor.execute(
        """SELECT progress_bars.version
        FROM progress_bars
        WHERE 
            EXISTS (
                SELECT 1 FROM users
                WHERE users.sub = ?
                  AND progress_bars.user_id = users.id
            )
            AND progress_bars.name = ?""",
        (user_sub, pbar_name),
    )
    if not response.results:
        return None
    return BarInfo(version=response.results[0][0])


async def get_steps_info(
    itgs: Itgs, user_sub: str, pbar_name: str, pbar_version: int
) -> List[StepInfo]:
    """gets the progress bar steps info from the database

    Args:
        itgs (Itgs): the integration to use
        user_sub (str): the user sub
        pbar_name (str): the name of the progress bar
        pbar_version (int): the version of the progress bar

    Returns:
        List[StepInfo]: the progress bar steps info
    """
    conn = await itgs.conn()
    cursor = conn.cursor("strong")
    response = await cursor.execute(
        """SELECT progress_bar_steps.name, progress_bar_steps.iterated
        FROM progress_bar_steps
        WHERE 
            EXISTS (
                SELECT 1 FROM users
                WHERE users.sub = ?
                  AND EXISTS (
                    SELECT 1 FROM progress_bars
                    WHERE progress_bars.user_id = users.id
                        AND progress_bars.id = progress_bar_steps.progress_bar_id
                        AND progress_bars.name = ?
                        AND progress_bars.version = ?
                    )
            )
            AND progress_bar_steps.position != 0
        ORDER BY progress_bar_steps.position
        """,
        (user_sub, pbar_name, pbar_version),
    )
    return [StepInfo(name=row[0], iterated=bool(row[1])) for row in response.results]


async def use_trace_to_initialize_pbar(
    itgs: Itgs,
    user_sub: str,
    pbar_name: str,
    trace_info: TraceInfo,
    steps_info: List[TraceStepInfo],
) -> None:
    """uses the trace info to initialize the progress bar

    Args:
        itgs (Itgs): the integration to use
        user_sub (str): the user sub
        pbar_name (str): the name of the progress bar
        trace_info (TraceInfo): the trace info
        steps_info (List[TraceStepInfo]): the trace steps info
    """
    conn = await itgs.conn()
    cursor = conn.cursor("strong")
    pbar_uid = "ep_pb_" + secrets.token_urlsafe(8)
    default_step_uid = "ep_pbs_" + secrets.token_urlsafe(8)
    step_uids = ["ep_pbs_" + secrets.token_urlsafe(8) for _ in range(len(steps_info))]
    now = time.time()
    await cursor.executemany3(
        (
            (
                """
                INSERT INTO progress_bars (
                    user_id,
                    uid,
                    name,
                    sampling_max_count,
                    sampling-max_age_seconds,
                    sampling_technique,
                    version,
                    created_at,
                )
                SELECT users.id, ?, ?, ?, ?, ?, ?, ?
                FROM users
                WHERE users.sub = ?
                    AND NOT EXISTS (
                        SELECT 1 FROM progress_bars AS progress_bars_inner
                        WHERE
                            progress_bars_inner.user_id = users.id
                            AND progress_bars_inner.name = ?
                    )
                """,
                (
                    pbar_uid,
                    pbar_name,
                    100,
                    604800,
                    "systematic",
                    0,
                    now,
                    user_sub,
                    pbar_name,
                ),
            ),
            (
                """
                    INSERT INTO progress_bar_steps (
                        progress_bar_id,
                        uid,
                        name,
                        position,
                        iterated,
                        one_off_technique,
                        one_off_percentile,
                        iterated_technique,
                        iterated_percentile,
                        created_at
                    )
                    SELECT
                        progress_bars.id,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?
                    FROM progress_bars
                    WHERE
                        progress_bars.uid = ?
                    """,
                (
                    default_step_uid,
                    "default",
                    0,
                    0,
                    "percentile",
                    75,
                    "best_fit.linear",
                    75,
                    now,
                    pbar_uid,
                ),
            ),
            *[
                (
                    """
                    INSERT INTO progress_bar_steps (
                        progress_bar_id,
                        uid,
                        name,
                        position,
                        iterated,
                        one_off_technique,
                        one_off_percentile,
                        iterated_technique,
                        iterated_percentile,
                        created_at
                    )
                    SELECT
                        progress_bars.id,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?
                    FROM progress_bars
                    WHERE
                        progress_bars.uid = ?
                    """,
                    (
                        step_uid,
                        step_info.step_name,
                        idx + 1,
                        int(step_info.iterations is not None),
                        "percentile",
                        75,
                        "best_fit.linear",
                        75,
                        now,
                        pbar_uid,
                    ),
                )
                for idx, (step_info, step_uid) in enumerate(zip(steps_info, step_uids))
            ],
        )
    )


async def use_trace_to_increment_pbar_version(
    itgs: Itgs,
    user_sub: str,
    pbar_name: str,
    pbar_version: int,
    trace_info: TraceInfo,
    steps_info: List[TraceStepInfo],
) -> None:
    """uses the trace info to increment the progress bar version

    Args:
        itgs (Itgs): the integration to use
        user_sub (str): the user sub
        pbar_name (str): the name of the progress bar
        pbar_version (int): the version of the progress bar
        trace_info (TraceInfo): the trace info
        steps_info (List[TraceStepInfo]): the trace steps info
    """
    conn = await itgs.conn()
    cursor = conn.cursor("strong")
    default_step_uid = "ep_pbs_" + secrets.token_urlsafe(8)
    step_uids = ["ep_pbs_" + secrets.token_urlsafe(8) for _ in range(len(steps_info))]
    now = time.time()
    await cursor.executemany3(
        (
            (
                """
                DELETE FROM progress_bar_traces
                WHERE
                    EXISTS (
                        SELECT 1 FROM progress_bars
                        WHERE progress_bars.id = progress_bar_traces.progress_bar_id
                            AND progress_bars.name = ?
                            AND progress_bars.version = ?
                            AND EXISTS (
                                SELECT 1 FROM users
                                WHERE users.sub = ?
                                  AND users.id = progress_bars.user_id
                            )
                    )
                """,
                (pbar_name, pbar_version, user_sub),
            ),
            (
                """
                DELETE FROM progress_bar_steps
                WHERE
                    EXISTS (
                        SELECT 1 FROM progress_bars
                        WHERE progress_bars.id = progress_bar_steps.progress_bar_id
                            AND progress_bars.name = ?
                            AND progress_bars.version = ?
                            AND EXISTS (
                                SELECT 1 FROM users
                                WHERE users.sub = ?
                                  AND users.id = progress_bars.user_id
                            )
                    )
                    """,
                (pbar_name, pbar_version, user_sub),
            ),
            (
                """
                    INSERT INTO progress_bar_steps (
                        progress_bar_id,
                        uid,
                        name,
                        position,
                        iterated,
                        one_off_technique,
                        one_off_percentile,
                        iterated_technique,
                        iterated_percentile,
                        created_at
                    )
                    SELECT
                        progress_bars.id,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?
                    FROM progress_bars
                    WHERE
                        EXISTS (
                            SELECT 1 FROM users
                            WHERE users.sub = ?
                                AND users.id = progress_bars.user_id
                        )
                        AND progress_bars.name = ?
                        AND progress_bars.version = ?

                    """,
                (
                    default_step_uid,
                    "default",
                    0,
                    0,
                    "percentile",
                    75,
                    "best_fit.linear",
                    75,
                    now,
                    user_sub,
                    pbar_name,
                    pbar_version,
                ),
            ),
            *[
                (
                    """
                    INSERT INTO progress_bar_steps (
                        progress_bar_id,
                        uid,
                        name,
                        position,
                        iterated,
                        one_off_technique,
                        one_off_percentile,
                        iterated_technique,
                        iterated_percentile,
                        created_at
                    )
                    SELECT
                        progress_bars.id,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?
                    FROM progress_bars
                    WHERE
                        EXISTS (
                            SELECT 1 FROM users
                            WHERE users.sub = ?
                                AND users.id = progress_bars.user_id
                        )
                        AND progress_bars.name = ?
                        AND progress_bars.version = ?
                    """,
                    (
                        step_uid,
                        step_info.step_name,
                        idx + 1,
                        int(step_info.iterations is not None),
                        "percentile",
                        75,
                        "best_fit.linear",
                        75,
                        now,
                        user_sub,
                        pbar_name,
                        pbar_version,
                    ),
                )
                for idx, (step_info, step_uid) in enumerate(zip(steps_info, step_uids))
            ],
            (
                """
                UPDATE progress_bars
                SET
                    progress_bars.version = progress_bars.version + 1
                WHERE
                    EXISTS (
                        SELECT 1 FROM users
                        WHERE users.sub = ?
                            AND users.id = progress_bars.user_id
                    )
                    AND progress_bars.name = ?
                    AND progress_bars.version = ?
                """,
                (user_sub, pbar_name, pbar_version),
            ),
        )
    )


async def sample_trace(
    itgs: Itgs,
    user_sub: str,
    pbar_name: str,
    pbar_version: int,
    trace_info: TraceInfo,
    steps_info: List[TraceStepInfo],
) -> None:
    """samples the trace info,if it should be sampled according to the progress
    bar sampling technique

    Args:
        itgs (Itgs): the integration to use
        user_sub (str): the user sub
        pbar_name (str): the name of the progress bar
        pbar_version (int): the version of the progress bar
        trace_info (TraceInfo): the trace info
        steps_info (List[TraceStepInfo]): the trace steps info"""

    conn = await itgs.conn()
    cursor = conn.cursor("weak")
    response = await cursor.execute(
        """
            SELECT
                progress_bars.sampling_max_count,
                progress_bars.sampling_max_age_seconds,
                progress_bars.sampling_technique
            FROM progress_bars
            WHERE
                EXISTS (
                    SELECT 1 FROM users
                    WHERE users.sub = ?
                        AND users.id = progress_bars.user_id
                )
                AND progress_bars.name = ?
                AND progress_bars.version = ?
            """,
        (user_sub, pbar_name, pbar_version),
    )
    if not response.results:
        return
    sampling_max_count: int = response.results[0][0]
    sampling_max_age_seconds: int = response.results[0][1] or 86400
    sampling_technique: str = response.results[0][2]
    if sampling_technique == "systematic":
        sample_interval: float = sampling_max_age_seconds / sampling_max_count
        trace_uid = "ep_pbt_" + secrets.token_urlsafe(8)
        tstep_uids = [
            "ep_pbts_" + secrets.token_urlsafe(8) for _ in range(len(steps_info))
        ]
        sampling_sql = """
            AND NOT EXISTS (
                SELECT 1 FROM progress_bar_traces
                WHERE
                    progress_bar_traces.progress_bar_id = progress_bars.id
                    AND progress_bar_traces.created_at > ?
            )
            """
        sampling_args = [trace_info.created_at - sample_interval]
    elif sampling_technique == "simple_random":
        redis = await itgs.redis()
        num_over_last_rolling_interval = await redis.zcount(
            f"tcount:{user_sub}:{pbar_name}:{pbar_version}",
            trace_info.created_at - sampling_max_age_seconds,
            "+inf",
        )
        if num_over_last_rolling_interval > sampling_max_count:
            sampling_chance = sampling_max_count / num_over_last_rolling_interval
            if random.random() > sampling_chance:
                return

        sampling_sql = ""
        sampling_args = []

    await cursor.executemany3(
        (
            (
                f"""
                INSERT INTO progress_bar_traces (
                    progress_bar_id,
                    uid,
                    created_at
                )
                SELECT
                    progress_bars.id,
                    ?, ?
                FROM progress_bars
                WHERE
                    EXISTS (
                        SELECT 1 FROM users
                        WHERE users.sub = ?
                            AND users.id = progress_bars.user_id
                    )
                    AND progress_bars.name = ?
                    AND progress_bars.version = ?
                    {sampling_sql}
                """,
                (
                    trace_uid,
                    trace_info.created_at,
                    user_sub,
                    pbar_name,
                    pbar_version,
                    *sampling_args,
                ),
            ),
            *[
                (
                    """
                    INSERT INTO progress_bar_trace_steps (
                        progress_bar_trace_id,
                        progress_bar_step_id,
                        uid,
                        iterations,
                        started_at,
                        finished_at
                    )
                    SELECT
                        progress_bar_traces.id,
                        progress_bar_steps.id,
                        ?, ?, ?, ?
                    FROM progress_bar_traces
                    JOIN progress_bar_steps ON progress_bar_steps.progress_bar_id = progress_bar_traces.progress_bar_id
                    WHERE
                        EXISTS (
                            SELECT 1 FROM progress_bars
                            WHERE
                                EXISTS (
                                    SELECT 1 FROM users
                                    WHERE users.sub = ?
                                        AND users.id = progress_bars.user_id
                                )
                                AND progress_bars.name = ?
                                AND progress_bars.version = ?
                                AND progress_bars.id = progress_bar_traces.progress_bar_id
                        )
                        AND progress_bar_traces.uid = ?
                        AND progress_bar_steps.position = ?
                    """,
                    (
                        tstep_uid,
                        step_info.iterations,
                        step_info.started_at,
                        step_info.finished_at,
                        user_sub,
                        pbar_name,
                        pbar_version,
                        trace_uid,
                        idx + 1,
                    ),
                )
                for idx, (step_info, tstep_uid) in enumerate(
                    zip(steps_info, tstep_uids)
                )
            ],
            (
                """
                DELETE FROM progress_bar_traces
                WHERE
                    EXISTS (
                        SELECT 1 FROM progress_bars
                        WHERE
                            EXISTS (
                                SELECT 1 FROM users
                                WHERE users.sub = ?
                                    AND users.id = progress_bars.user_id
                            )
                            AND progress_bars.name = ?
                            AND progress_bars.version = ?
                            AND progress_bars.id = progress_bar_traces.progress_bar_id
                    )
                    AND progress_bar_traces.created_at < ? - progress_bars.sampling_max_age_seconds
                """,
                (user_sub, pbar_name, pbar_version, trace_info.created_at),
            ),
        )
    )


async def update_tcount(
    itgs: Itgs,
    user_sub: str,
    pbar_name: str,
    pbar_version: int,
    trace_uid: str,
    trace_created_at: float,
) -> None:
    """updates the tcount for the given progress bar

    Args:
        itgs (Itgs): the integration to use
        user_sub (str): the user sub
        pbar_name (str): the name of the progress bar
        pbar_version (int): the version of the progress bar
        trace_created_at (float): the trace created at"""
    conn = await itgs.conn()
    cursor = conn.cursor("weak")
    response = await cursor.execute(
        """
        SELECT progress_bars.sampling_max_age_seconds
        FROM progress_bars
        WHERE
            EXISTS (
                SELECT 1 FROM users
                WHERE users.sub = ?
                    AND users.id = progress_bars.user_id
            )
            AND progress_bars.name = ?
            AND progress_bars.version = ?
        """,
        (user_sub, pbar_name, pbar_version),
    )
    if not response.results:
        return
    sampling_max_age_seconds: int = response.results[0][0]

    redis = await itgs.redis()
    key = f"tcount:{user_sub}:{pbar_name}:{pbar_version}"
    async with redis.pipeline() as pipe:
        pipe.multi()
        await pipe.zadd(
            key,
            {trace_uid: trace_created_at},
            nx=True,
        )
        await pipe.zremrangebyscore(
            key,
            "-inf",
            trace_created_at - sampling_max_age_seconds - 300,
        )
        await pipe.expire(key, sampling_max_age_seconds + 300)
        await pipe.execute()
