"""a very simple job"""
import asyncio
from typing import List, Literal, Optional, Tuple
from pydantic import BaseModel, Field
from itgs import Itgs
from graceful_death import GracefulDeath
from dataclasses import dataclass
import numpy as np
from numpy.polynomial import Polynomial
from scipy.stats.mstats import gmean, hmean
from redis.asyncio.client import Pipeline


EXCLUSIVE = True


class JobResultStepItem(BaseModel):
    step_name: str = Field(description="the name of the step")
    iterated: bool = Field(description="whether the step is iterated")
    technique: str = Field(description="the technique used to calculate the eta")
    percentile: Optional[float] = Field(
        description="the percentile if the technique is percentile"
    )
    eta_a: float = Field(
        description="the first variable in the fit; for all techniques except best_fit.linear this is the only variable required - for example, if the technique is arithmetic mean, this is the arithmetic mean. for best_fit.linear, this is the slope of the fit"
    )
    eta_b: Optional[float] = Field(
        description="the second variable of the fit; unused except for best_fit.linear, where this is the intercept of the fit"
    )


class JobResult(BaseModel):
    technique: str = Field(
        description="the technique used to calculate the overall eta, this is a one-off technique"
    )
    percentile: Optional[float] = Field(
        description="the percentile if the technique is percentile"
    )
    eta: float = Field(description="the eta calculated using the technique")
    steps: List[JobResultStepItem] = Field(
        description="the steps of the progress bar in order"
    )
    version: int = Field(
        description="the version of the progress bar at the time of calculation"
    )


@dataclass
class BarInfo:
    user_sub: str
    uid: str
    name: str
    version: int


@dataclass
class BarStepInfo:
    uid: str
    name: str
    iterated: bool
    one_off_technique: Literal[
        "percentile", "arithmetic_mean", "geometric_mean", "harmonic_mean"
    ]
    one_off_percentile: float
    iterated_technique: Literal[
        "best_fit.linear",
        "percentile",
        "arithmetic_mean",
        "geometric_mean",
        "harmonic_mean",
    ]
    iterated_percentile: float


async def execute(
    itgs: Itgs, gd: GracefulDeath, *, user_sub: str, pbar_name: str, job_uid: str
):
    """updates the estimated time to completion on the current progress bar
    based on the currently stored traces

    Args:
        itgs (Itgs): the integration to use; provided automatically
        gd (GracefulDeath): the signal tracker; provided automatically
        user_sub (str): the user sub
        pbar_name (str): the name of the progress bar
        job_uid (str): the job uid: we publish a message when the job is done with the above job result
    """

    async def bounce():
        jobs = await itgs.jobs()
        await jobs.enqueue(
            "runners.calc_pbar_eta",
            user_sub=user_sub,
            pbar_name=pbar_name,
            job_uid=job_uid,
        )

    redis = await itgs.redis()
    bar_info = await get_bar_info(itgs, user_sub=user_sub, pbar_name=pbar_name)
    if bar_info is None:
        await redis.publish(
            f"ps:job:{job_uid}",
            JobResult(
                technique="percentile",
                percentile=0.5,
                eta=0,
                steps=[],
                version=-1,
            ).json(),
        )
        return
    if gd.received_term_signal:
        return await bounce()
    steps_info = await get_bar_step_info(itgs, bar_info=bar_info)
    if steps_info is None:
        await redis.publish(
            f"ps:job:{job_uid}",
            JobResult(
                technique="percentile",
                percentile=0.5,
                eta=0,
                steps=[],
                version=-1,
            ).json(),
        )
        return
    if gd.received_term_signal:
        return await bounce()
    overall_eta = await {
        "percentile": get_overall_eta_percentile,
        "arithmetic_mean": get_overall_eta_arithmetic_mean,
        "geometric_mean": get_overall_eta_geometric_mean,
        "harmonic_mean": get_overall_eta_harmonic_mean,
    }[steps_info[0].one_off_technique](itgs, bar_info, steps_info)
    if overall_eta is None:
        await redis.publish(
            f"ps:job:{job_uid}",
            JobResult(
                technique=steps_info[0].one_off_technique,
                percentile=steps_info[0].one_off_percentile
                if steps_info[0].one_off_technique == "percentile"
                else None,
                eta=0,
                steps=[
                    JobResultStepItem(
                        step_name=step_info.name,
                        iterated=step_info.iterated,
                        technique="percentile",
                        percentile=75,
                        eta_a=0,
                        eta_b=None,
                    )
                    for step_info in steps_info
                ],
                version=bar_info.version,
            ).json(),
        )
        return
    if gd.received_term_signal:
        return await bounce()
    step_eta_funcs = {
        (False, "percentile"): get_step_eta_one_off_percentile,
        (False, "arithmetic_mean"): get_step_eta_one_off_arithmetic_mean,
        (False, "geometric_mean"): get_step_eta_one_off_geometric_mean,
        (False, "harmonic_mean"): get_step_eta_one_off_harmonic_mean,
        (True, "best_fit.linear"): get_step_eta_iterated_best_fit_linear,
        (True, "percentile"): get_step_eta_iterated_percentile,
        (True, "arithmetic_mean"): get_step_eta_iterated_arithmetic_mean,
        (True, "geometric_mean"): get_step_eta_iterated_geometric_mean,
        (True, "harmonic_mean"): get_step_eta_iterated_harmonic_mean,
    }
    step_etas = await asyncio.gather(
        *[
            step_eta_funcs[
                (
                    step_info.iterated,
                    step_info.iterated_technique
                    if step_info.iterated
                    else step_info.one_off_technique,
                )
            ](itgs, bar_info, steps_info, idx)
            for idx, step_info in enumerate(steps_info)
            if idx != 0
        ],
        return_exceptions=True,
    )
    for step_eta in step_etas:
        if isinstance(step_eta, Exception):
            raise step_eta
    step_etas = [
        calc_eta
        if not isinstance(calc_eta, (Exception, type(None)))
        else (
            0,
            None
            if not step_info.iterated
            or step_info.iterated_technique != "best_fit.linear"
            else 0,
        )
        for calc_eta, step_info in zip(step_etas, steps_info[1:])
    ]
    result = JobResult(
        technique=steps_info[0].one_off_technique,
        percentile=steps_info[0].one_off_percentile
        if steps_info[0].one_off_technique == "percentile"
        else None,
        eta=overall_eta,
        steps=[
            JobResultStepItem(
                step_name=step_info.name,
                iterated=step_info.iterated,
                technique=step_info.one_off_technique
                if not step_info.iterated
                else step_info.iterated_technique,
                percentile=(
                    (
                        step_info.one_off_percentile
                        if step_info.one_off_technique == "percentile"
                        else None
                    )
                    if not step_info.iterated
                    else (
                        step_info.iterated_percentile
                        if step_info.iterated_technique == "percentile"
                        else None
                    )
                ),
                eta_a=eta_a,
                eta_b=eta_b,
            )
            for step_info, (eta_a, eta_b) in zip(steps_info, step_etas)
        ],
        version=bar_info.version,
    )
    await redis.publish(f"ps:job:{job_uid}", result.json())
    async with await redis.pipeline(True) as pipe:
        pipe: Pipeline
        technique = result.technique
        if technique == "percentile":
            technique += f"_{result.percentile}"
        pipe.multi()
        await pipe.set(
            f"stats:{user_sub}:{pbar_name}:{bar_info.version}:{technique}",
            result.eta,
        )
        await pipe.expire(
            f"stats:{user_sub}:{pbar_name}:{bar_info.version}:{technique}",
            60 * 30,
        )
        for idx, step in enumerate(result.steps):
            if idx == 0:
                continue
            technique = step.technique
            if technique == "percentile":
                technique += f"_{step.percentile}"
            await pipe.hset(
                f"stats:{user_sub}:{pbar_name}:{bar_info.version}:{idx}:{technique}",
                mapping=dict(
                    (k, v)
                    for k, v in {
                        "a": step.eta_a,
                        "b": step.eta_b,
                    }.items()
                    if v is not None
                ),
            )
            await pipe.expire(
                f"stats:{user_sub}:{pbar_name}:{bar_info.version}:{idx}:{technique}",
                60 * 30,
            )
        await pipe.execute()


async def get_bar_info(
    itgs: Itgs, *, user_sub: str, pbar_name: str
) -> Optional[BarInfo]:
    """gets the bar info for the given progress bar

    Args:
        itgs (Itgs): the integration to use; provided automatically
        user_sub (str): the user sub
        pbar_name (str): the name of the progress bar

    Returns:
        BarInfo, None: the bar info if such a bar exists
    """
    conn = await itgs.conn()
    cursor = conn.cursor("none")
    response = await cursor.execute(
        """
        SELECT progress_bars.uid, progress_bars.version
        FROM progress_bars
        WHERE
            EXISTS (
                SELECT 1 FROM users
                WHERE users.sub = ?
                AND users.id = progress_bars.user_id
            )
            AND progress_bars.name = ?
        """,
        (user_sub, pbar_name),
    )
    if not response.results:
        return None

    return BarInfo(
        user_sub=user_sub,
        uid=response.results[0][0],
        name=pbar_name,
        version=response.results[0][1],
    )


async def get_bar_step_info(
    itgs: Itgs, bar_info: BarInfo
) -> Optional[List[BarStepInfo]]:
    """
    gets the steps for the given progress bar, where index 0 corresponds to position 0 i.e., the default step

    Args:
        itgs (Itgs): the integrations to use
        bar_info (BarInfo): the progress bar to get the steps for
    """
    conn = await itgs.conn()
    cursor = conn.cursor("none")
    response = await cursor.execute(
        """
        SELECT 
            progress_bar_steps.uid,
            progress_bar_steps.name,
            progress_bar_steps.iterated,
            progress_bar_steps.one_off_technique,
            progress_bar_steps.one_off_percentile,
            progress_bar_steps.iterated_technique,
            progress_bar_steps.iterated_percentile
        FROM progress_bar_steps
        WHERE 
            EXISTS (
                SELECT 1 FROM progress_bars
                WHERE progress_bars.id = progress_bar_steps.progress_bar_id
                AND EXISTS (
                    SELECT 1 FROM users
                    WHERE users.sub = ?
                    AND users.id = progress_bars.user_id
                )
                AND progress_bars.name = ?
                AND progress_bars.version = ?
            )
        ORDER BY progress_bar_steps.position ASC
        """,
        (bar_info.user_sub, bar_info.name, bar_info.version),
    )
    if not response.results:
        return None
    return [
        BarStepInfo(
            uid=step[0],
            name=step[1],
            iterated=bool(step[2]),
            one_off_technique=step[3],
            one_off_percentile=step[4],
            iterated_technique=step[5],
            iterated_percentile=step[6],
        )
        for step in response.results
    ]


async def get_overall_eta_percentile(
    itgs: Itgs, bar_info: BarInfo, steps_info: List[BarStepInfo]
) -> Optional[float]:
    """gets the overall eta percentile for the given progress bar

    Args:
        itgs (Itgs): the integrations to use
        bar_info (BarInfo): the progress bar to get the overall eta percentile for
        steps_info (List[BarStepInfo]): the steps for the progress bar

    Returns:
        Optional[float]: the overall eta percentile if the progress bar exists and has traces
    """
    conn = await itgs.conn()
    cursor = conn.cursor("none")
    response = await cursor.execute(
        """
        WITH trace_counts AS (
            SELECT
                progress_bar_traces.progress_bar_id AS progress_bar_id,
                COUNT(*) AS trace_count
            FROM progress_bar_traces
            GROUP BY progress_bar_traces.progress_bar_id
        ),
        trace_durations AS (
            SELECT
                progress_bar_traces.id AS trace_id,
                progress_bar_traces.progress_bar_id AS progress_bar_id,
                progress_bar_trace_steps.finished_at - progress_bar_traces.created_at AS duration
            FROM progress_bar_traces
            JOIN progress_bar_trace_steps ON progress_bar_trace_steps.progress_bar_trace_id = progress_bar_traces.id
            WHERE EXISTS (
                SELECT 1 FROM progress_bar_steps
                WHERE progress_bar_steps.id = progress_bar_trace_steps.progress_bar_step_id
                AND progress_bar_steps.position = ?
            )
        )
        SELECT
            trace_durations.duration
        FROM trace_durations
        WHERE
            EXISTS (
                SELECT 1 FROM progress_bars
                WHERE progress_bars.id = trace_durations.progress_bar_id
                AND progress_bars.uid = ?
                AND progress_bars.version = ?
            )
        ORDER BY trace_durations.duration ASC
        LIMIT 1 OFFSET (
            SELECT
                COALESCE(AVG(CAST(COALESCE(trace_counts.trace_count, 0) * ? AS INTEGER)),0)
            FROM trace_counts
            WHERE EXISTS (
                SELECT 1 FROM progress_bars
                WHERE progress_bars.id = trace_counts.progress_bar_id
                AND progress_bars.uid = ?
                AND progress_bars.version = ?
            )
        )
        """,
        (
            len(steps_info) - 1,
            bar_info.uid,
            bar_info.version,
            steps_info[0].one_off_percentile / 100,
            bar_info.uid,
            bar_info.version,
        ),
    )
    if not response.results:
        return None
    return response.results[0][0]


async def get_overall_eta_arithmetic_mean(
    itgs: Itgs, bar_info: BarInfo, steps_info: List[BarStepInfo]
) -> Optional[float]:
    """
    gets the overall eta using the arithmetic mean for the given progress bar

    Args:
        itgs (Itgs): the integrations to use
        bar_info (BarInfo): the progress bar to get the overall eta percentile for
        steps_info (List[BarStepInfo]): the steps for the progress bar

    Returns:
        Optional[float]: the overall eta arithmetic mean if the progress bar exists and has traces
    """
    conn = await itgs.conn()
    cursor = conn.cursor("none")
    response = await cursor.execute(
        """
        WITH trace_durations AS (
            SELECT
                progress_bar_traces.id AS trace_id,
                progress_bar_traces.progress_bar_id AS progress_bar_id,
                progress_bar_trace_steps.finished_at - progress_bar_traces.created_at AS duration
            FROM progress_bar_traces
            JOIN progress_bar_trace_steps ON progress_bar_trace_steps.progress_bar_trace_id = progress_bar_traces.id
            WHERE
                EXISTS (
                    SELECT 1 FROM progress_bar_steps
                    WHERE progress_bar_steps.id = progress_bar_trace_steps.progress_bar_step_id
                    AND progress_bar_steps.position = ?
                )
        )
        SELECT
            AVG(trace_durations.duration)
        FROM trace_durations
        WHERE
            EXISTS (
                SELECT 1 FROM progress_bars
                WHERE progress_bars.id = trace_durations.progress_bar_id
                AND progress_bars.uid = ?
                AND progress_bars.version = ?
            )
        """,
        (
            len(steps_info) - 1,
            bar_info.uid,
            bar_info.version,
        ),
    )
    if not response.results or response.results[0][0] is None:
        return None
    return response.results[0][0]


async def get_overall_eta_geometric_mean(
    itgs: Itgs, bar_info: BarInfo, steps_info: List[BarStepInfo]
) -> Optional[float]:
    """
    gets the overall eta using the geometric mean for the given progress bar

    Args:
        itgs (Itgs): the integrations to use
        bar_info (BarInfo): the progress bar to get the overall eta percentile for
        steps_info (List[BarStepInfo]): the steps for the progress bar

    Returns:
        Optional[float]: the overall eta geometric mean if the progress bar exists and has traces
    """
    conn = await itgs.conn()
    cursor = conn.cursor("none")
    response = await cursor.execute(
        """
        WITH trace_durations AS (
            SELECT
                progress_bar_traces.id AS trace_id,
                progress_bar_traces.progress_bar_id AS progress_bar_id,
                progress_bar_trace_steps.finished_at - progress_bar_traces.created_at AS duration
            FROM progress_bar_traces
            JOIN progress_bar_trace_steps ON progress_bar_trace_steps.progress_bar_trace_id = progress_bar_traces.id
            WHERE
                EXISTS (
                    SELECT 1 FROM progress_bar_steps
                    WHERE progress_bar_steps.id = progress_bar_trace_steps.progress_bar_step_id
                    AND progress_bar_steps.position = ?
                )
        )
        SELECT
            trace_durations.duration
        FROM trace_durations
        WHERE
            EXISTS (
                SELECT 1 FROM progress_bars
                WHERE progress_bars.id = trace_durations.progress_bar_id
                AND progress_bars.uid = ?
                AND progress_bars.version = ?
            )
        """,
        (
            len(steps_info) - 1,
            bar_info.uid,
            bar_info.version,
        ),
    )
    if not response.results:
        return None
    return float(gmean(np.array(response.results).reshape(len(response.results))))


async def get_overall_eta_harmonic_mean(
    itgs: Itgs, bar_info: BarInfo, steps_info: List[BarStepInfo]
) -> Optional[float]:
    """
    gets the overall eta using the harmonic mean for the given progress bar

    Args:
        itgs (Itgs): the integrations to use
        bar_info (BarInfo): the progress bar to get the overall eta percentile for
        steps_info (List[BarStepInfo]): the steps for the progress bar

    Returns:
        Optional[float]: the overall eta harmonic mean if the progress bar exists and has traces
    """
    conn = await itgs.conn()
    cursor = conn.cursor("none")
    response = await cursor.execute(
        """
        WITH trace_durations AS (
            SELECT
                progress_bar_traces.id AS trace_id,
                progress_bar_traces.progress_bar_id AS progress_bar_id,
                progress_bar_trace_steps.finished_at - progress_bar_traces.created_at AS duration
            FROM progress_bar_traces
            JOIN progress_bar_trace_steps ON progress_bar_trace_steps.progress_bar_trace_id = progress_bar_traces.id
            WHERE
                EXISTS (
                    SELECT 1 FROM progress_bar_steps
                    WHERE progress_bar_steps.id = progress_bar_trace_steps.progress_bar_step_id
                    AND progress_bar_steps.position = ?
                )
        )
        SELECT
            trace_durations.duration
        FROM trace_durations
        WHERE
            EXISTS (
                SELECT 1 FROM progress_bars
                WHERE progress_bars.id = trace_durations.progress_bar_id
                AND progress_bars.uid = ?
                AND progress_bars.version = ?
            )
        """,
        (
            len(steps_info) - 1,
            bar_info.uid,
            bar_info.version,
        ),
    )
    if not response.results:
        return None
    return float(hmean(np.array(response.results).reshape(len(response.results))))


async def get_step_eta_one_off_percentile(
    itgs: Itgs, bar_info: BarInfo, steps_info: List[BarStepInfo], step_position: int
) -> Optional[Tuple[float, Optional[float]]]:
    """
    gets the eta for the given one-off step for the given progress bar using the percentile technique

    Args:
        itgs (Itgs): the integrations to use
        bar_info (BarInfo): the progress bar to get the step eta for
        steps_info (List[BarStepInfo]): the steps for the progress bar
        step_position (int): the position of the step to get the eta for

    Returns:
        Optional[Tuple[float, Optional[float]]]: the step eta percentile if the progress bar exists and has traces
    """
    conn = await itgs.conn()
    cursor = conn.cursor("none")
    response = await cursor.execute(
        """
        WITH trace_counts AS (
            SELECT
                progress_bar_traces.progress_bar_id AS progress_bar_id,
                COUNT(*) AS trace_count
            FROM progress_bar_traces
            GROUP BY progress_bar_traces.progress_bar_id
        ),
        trace_step_durations AS (
            SELECT
                progress_bar_trace_steps.progress_bar_trace_id AS trace_id,
                progress_bar_trace_steps.progress_bar_step_id AS progress_bar_step_id,
                progress_bar_traces.progress_bar_id AS progress_bar_id,
                progress_bar_steps.position AS step_position,
                progress_bar_trace_steps.finished_at - progress_bar_trace_steps.started_at AS duration
            FROM progress_bar_trace_steps
            JOIN progress_bar_traces ON progress_bar_traces.id = progress_bar_trace_steps.progress_bar_trace_id
            JOIN progress_bar_steps ON progress_bar_steps.id = progress_bar_trace_steps.progress_bar_step_id
        )
        SELECT
            trace_step_durations.duration
        FROM trace_step_durations
        WHERE
            EXISTS (
                SELECT 1 FROM progress_bars
                WHERE progress_bars.id = trace_step_durations.progress_bar_id
                AND progress_bars.uid = ?
                AND progress_bars.version = ?
            )
            AND trace_step_durations.step_position = ?
        ORDER BY trace_step_durations.duration
        LIMIT 1 OFFSET (
            SELECT
                COALESCE(AVG(CAST(COALESCE(trace_counts.trace_count, 0) * ? AS INTEGER)),0)
            FROM trace_counts
            WHERE EXISTS (
                SELECT 1 FROM progress_bars
                WHERE progress_bars.id = trace_counts.progress_bar_id
                AND progress_bars.uid = ?
                AND progress_bars.version = ?
            )
        )
        """,
        (
            bar_info.uid,
            bar_info.version,
            step_position,
            steps_info[step_position].one_off_percentile / 100,
            bar_info.uid,
            bar_info.version,
        ),
    )
    if not response.results:
        return None
    return response.results[0][0], None


async def get_step_eta_one_off_arithmetic_mean(
    itgs: Itgs, bar_info: BarInfo, steps_info: List[BarStepInfo], step_position: int
) -> Optional[Tuple[float, Optional[float]]]:
    """
    gets the eta for the given one-off step for the given progress bar using the arithmetic mean technique

    Args:
        itgs (Itgs): the integrations to use
        bar_info (BarInfo): the progress bar to get the step eta for
        steps_info (List[BarStepInfo]): the steps for the progress bar
        step_position (int): the position of the step to get the eta for

    Returns:
        Optional[Tuple[float, Optional[float]]]: the step eta arithmetic mean if the progress bar exists and has traces
    """
    conn = await itgs.conn()
    cursor = conn.cursor("none")
    response = await cursor.execute(
        """
        WITH trace_step_durations AS (
            SELECT
                progress_bar_traces.progress_bar_id AS progress_bar_id,
                progress_bar_steps.position AS step_position,
                progress_bar_trace_steps.finished_at - progress_bar_trace_steps.started_at AS duration
            FROM progress_bar_trace_steps
            JOIN progress_bar_traces ON progress_bar_traces.id = progress_bar_trace_steps.progress_bar_trace_id
            JOIN progress_bar_steps ON progress_bar_steps.id = progress_bar_trace_steps.progress_bar_step_id
        )
        SELECT
            AVG(trace_step_durations.duration)
        FROM trace_step_durations
        WHERE
            EXISTS (
                SELECT 1 FROM progress_bars
                WHERE progress_bars.id = trace_step_durations.progress_bar_id
                AND progress_bars.uid = ?
                AND progress_bars.version = ?
            )
            AND trace_step_durations.step_position = ?
        """,
        (
            bar_info.uid,
            bar_info.version,
            step_position,
        ),
    )
    if not response.results or response.results[0][0] is None:
        return None
    return response.results[0][0], None


async def get_step_eta_one_off_harmonic_mean(
    itgs: Itgs, bar_info: BarInfo, steps_info: List[BarStepInfo], step_position: int
) -> Optional[Tuple[float, Optional[float]]]:
    """
    gets the eta for the given one-off step for the given progress bar using the harmonic mean technique

    Args:
        itgs (Itgs): the integrations to use
        bar_info (BarInfo): the progress bar to get the step eta for
        steps_info (List[BarStepInfo]): the steps for the progress bar
        step_position (int): the position of the step to get the eta for

    Returns:
        Optional[Tuple[float, Optional[float]]]: the step eta harmonic mean if the progress bar exists and has traces
    """
    conn = await itgs.conn()
    cursor = conn.cursor("none")
    response = await cursor.execute(
        """
        WITH trace_step_durations AS (
            SELECT
                progress_bar_traces.progress_bar_id AS progress_bar_id,
                progress_bar_steps.position AS step_position,
                progress_bar_trace_steps.finished_at - progress_bar_trace_steps.started_at AS duration
            FROM progress_bar_trace_steps
            JOIN progress_bar_traces ON progress_bar_traces.id = progress_bar_trace_steps.progress_bar_trace_id
            JOIN progress_bar_steps ON progress_bar_steps.id = progress_bar_trace_steps.progress_bar_step_id
        )
        SELECT
            trace_step_durations.duration
        FROM trace_step_durations
        WHERE
            EXISTS (
                SELECT 1 FROM progress_bars
                WHERE progress_bars.id = trace_step_durations.progress_bar_id
                AND progress_bars.uid = ?
                AND progress_bars.version = ?
            )
            AND trace_step_durations.step_position = ?
        """,
        (
            bar_info.uid,
            bar_info.version,
            step_position,
        ),
    )
    if not response.results:
        return None
    return (
        float(hmean(np.array(response.results).reshape(len(response.results)))),
        None,
    )


async def get_step_eta_one_off_geometric_mean(
    itgs: Itgs, bar_info: BarInfo, steps_info: List[BarStepInfo], step_position: int
) -> Optional[Tuple[float, Optional[float]]]:
    """
    gets the eta for the given one-off step for the given progress bar using the geometric mean technique

    Args:
        itgs (Itgs): the integrations to use
        bar_info (BarInfo): the progress bar to get the step eta for
        steps_info (List[BarStepInfo]): the steps for the progress bar
        step_position (int): the position of the step to get the eta for

    Returns:
        Optional[Tuple[float, Optional[float]]]: the step eta geometric mean if the progress bar exists and has traces
    """
    conn = await itgs.conn()
    cursor = conn.cursor("none")
    response = await cursor.execute(
        """
        WITH trace_step_durations AS (
            SELECT
                progress_bar_traces.progress_bar_id AS progress_bar_id,
                progress_bar_steps.position AS step_position,
                progress_bar_trace_steps.finished_at - progress_bar_trace_steps.started_at AS duration
            FROM progress_bar_trace_steps
            JOIN progress_bar_traces ON progress_bar_traces.id = progress_bar_trace_steps.progress_bar_trace_id
            JOIN progress_bar_steps ON progress_bar_steps.id = progress_bar_trace_steps.progress_bar_step_id
        )
        SELECT
            trace_step_durations.duration
        FROM trace_step_durations
        WHERE
            EXISTS (
                SELECT 1 FROM progress_bars
                WHERE progress_bars.id = trace_step_durations.progress_bar_id
                AND progress_bars.uid = ?
                AND progress_bars.version = ?
            )
            AND trace_step_durations.step_position = ?
        """,
        (
            bar_info.uid,
            bar_info.version,
            step_position,
        ),
    )
    if not response.results:
        return None
    return (
        float(gmean(np.array(response.results).reshape(len(response.results)))),
        None,
    )


async def get_step_eta_iterated_percentile(
    itgs: Itgs, bar_info: BarInfo, steps_info: List[BarStepInfo], step_position: int
) -> Optional[Tuple[float, Optional[float]]]:
    """
    gets the eta for the given iterated step for the given progress bar using the percentile technique

    Args:
        itgs (Itgs): the integrations to use
        bar_info (BarInfo): the progress bar to get the step eta for
        steps_info (List[BarStepInfo]): the steps for the progress bar
        step_position (int): the position of the step to get the eta for

    Returns:
        Optional[Tuple[float, Optional[float]]]: the step eta percentile if the progress bar exists and has traces
    """
    conn = await itgs.conn()
    cursor = conn.cursor("none")
    response = await cursor.execute(
        """
        WITH trace_counts AS (
            SELECT
                progress_bar_traces.progress_bar_id AS progress_bar_id,
                COUNT(*) AS trace_count
            FROM progress_bar_traces
            GROUP BY progress_bar_traces.progress_bar_id
        ),
        trace_step_durations AS (
            SELECT
                progress_bar_trace_steps.progress_bar_trace_id AS trace_id,
                progress_bar_trace_steps.progress_bar_step_id AS progress_bar_step_id,
                progress_bar_traces.progress_bar_id AS progress_bar_id,
                progress_bar_steps.position AS step_position,
                (progress_bar_trace_steps.finished_at - progress_bar_trace_steps.started_at) / progress_bar_trace_steps.iterations AS duration
            FROM progress_bar_trace_steps
            JOIN progress_bar_traces ON progress_bar_traces.id = progress_bar_trace_steps.progress_bar_trace_id
            JOIN progress_bar_steps ON progress_bar_steps.id = progress_bar_trace_steps.progress_bar_step_id
        )
        SELECT
            trace_step_durations.duration
        FROM trace_step_durations
        WHERE
            EXISTS (
                SELECT 1 FROM progress_bars
                WHERE progress_bars.id = trace_step_durations.progress_bar_id
                AND progress_bars.uid = ?
                AND progress_bars.version = ?
            )
            AND trace_step_durations.step_position = ?
        ORDER BY trace_step_durations.duration
        LIMIT 1 OFFSET (
            SELECT
                COALESCE(AVG(CAST(COALESCE(trace_counts.trace_count, 0) * ? AS INTEGER)),0)
            FROM trace_counts
            WHERE EXISTS (
                SELECT 1 FROM progress_bars
                WHERE progress_bars.id = trace_counts.progress_bar_id
                AND progress_bars.uid = ?
                AND progress_bars.version = ?
            )
        )
        """,
        (
            bar_info.uid,
            bar_info.version,
            step_position,
            steps_info[step_position].iterated_percentile / 100,
            bar_info.uid,
            bar_info.version,
        ),
    )
    if not response.results:
        return None
    return response.results[0][0], None


async def get_step_eta_iterated_arithmetic_mean(
    itgs: Itgs, bar_info: BarInfo, steps_info: List[BarStepInfo], step_position: int
) -> Optional[Tuple[float, Optional[float]]]:
    """
    gets the eta for the given iterated step for the given progress bar using the arithmetic mean technique

    Args:
        itgs (Itgs): the integrations to use
        bar_info (BarInfo): the progress bar to get the step eta for
        steps_info (List[BarStepInfo]): the steps for the progress bar
        step_position (int): the position of the step to get the eta for

    Returns:
        Optional[Tuple[float, Optional[float]]]: the step eta arithmetic mean if the progress bar exists and has traces
    """
    conn = await itgs.conn()
    cursor = conn.cursor("none")
    response = await cursor.execute(
        """
        WITH trace_step_durations AS (
            SELECT
                progress_bar_traces.progress_bar_id AS progress_bar_id,
                progress_bar_steps.position AS step_position,
                (progress_bar_trace_steps.finished_at - progress_bar_trace_steps.started_at) / progress_bar_trace_steps.iterations AS duration
            FROM progress_bar_trace_steps
            JOIN progress_bar_traces ON progress_bar_traces.id = progress_bar_trace_steps.progress_bar_trace_id
            JOIN progress_bar_steps ON progress_bar_steps.id = progress_bar_trace_steps.progress_bar_step_id
        )
        SELECT
            AVG(trace_step_durations.duration)
        FROM trace_step_durations
        WHERE
            EXISTS (
                SELECT 1 FROM progress_bars
                WHERE progress_bars.id = trace_step_durations.progress_bar_id
                AND progress_bars.uid = ?
                AND progress_bars.version = ?
            )
            AND trace_step_durations.step_position = ?
        """,
        (
            bar_info.uid,
            bar_info.version,
            step_position,
        ),
    )
    if not response.results or response.results[0][0] is None:
        return None
    return response.results[0][0], None


async def get_step_eta_iterated_geometric_mean(
    itgs: Itgs, bar_info: BarInfo, steps_info: List[BarStepInfo], step_position: int
) -> Optional[Tuple[float, Optional[float]]]:
    """
    gets the eta for the given iterated step for the given progress bar using the geometric mean technique

    Args:
        itgs (Itgs): the integrations to use
        bar_info (BarInfo): the progress bar to get the step eta for
        steps_info (List[BarStepInfo]): the steps for the progress bar
        step_position (int): the position of the step to get the eta for

    Returns:
        Optional[Tuple[float, Optional[float]]]: the step eta geometric mean if the progress bar exists and has traces
    """
    conn = await itgs.conn()
    cursor = conn.cursor("none")
    response = await cursor.execute(
        """
        WITH trace_step_durations AS (
            SELECT
                progress_bar_traces.progress_bar_id AS progress_bar_id,
                progress_bar_steps.position AS step_position,
                (progress_bar_trace_steps.finished_at - progress_bar_trace_steps.started_at) / progress_bar_trace_steps.iterations AS duration
            FROM progress_bar_trace_steps
            JOIN progress_bar_traces ON progress_bar_traces.id = progress_bar_trace_steps.progress_bar_trace_id
            JOIN progress_bar_steps ON progress_bar_steps.id = progress_bar_trace_steps.progress_bar_step_id
        )
        SELECT
            trace_step_durations.duration
        FROM trace_step_durations
        WHERE
            EXISTS (
                SELECT 1 FROM progress_bars
                WHERE progress_bars.id = trace_step_durations.progress_bar_id
                AND progress_bars.uid = ?
                AND progress_bars.version = ?
            )
            AND trace_step_durations.step_position = ?
        """,
        (
            bar_info.uid,
            bar_info.version,
            step_position,
        ),
    )
    if not response.results:
        return None
    return (
        float(gmean(np.array(response.results).reshape(len(response.results)))),
        None,
    )


async def get_step_eta_iterated_harmonic_mean(
    itgs: Itgs, bar_info: BarInfo, steps_info: List[BarStepInfo], step_position: int
) -> Optional[Tuple[float, Optional[float]]]:
    """
    gets the eta for the given iterated step for the given progress bar using the harmonic mean technique

    Args:
        itgs (Itgs): the integrations to use
        bar_info (BarInfo): the progress bar to get the step eta for
        steps_info (List[BarStepInfo]): the steps for the progress bar
        step_position (int): the position of the step to get the eta for

    Returns:
        Optional[Tuple[float, Optional[float]]]: the step eta harmonic mean if the progress bar exists and has traces
    """
    conn = await itgs.conn()
    cursor = conn.cursor("none")
    response = await cursor.execute(
        """
        WITH trace_step_durations AS (
            SELECT
                progress_bar_traces.progress_bar_id AS progress_bar_id,
                progress_bar_steps.position AS step_position,
                (progress_bar_trace_steps.finished_at - progress_bar_trace_steps.started_at) / progress_bar_trace_steps.iterations AS duration
            FROM progress_bar_trace_steps
            JOIN progress_bar_traces ON progress_bar_traces.id = progress_bar_trace_steps.progress_bar_trace_id
            JOIN progress_bar_steps ON progress_bar_steps.id = progress_bar_trace_steps.progress_bar_step_id
        )
        SELECT
            trace_step_durations.duration
        FROM trace_step_durations
        WHERE
            EXISTS (
                SELECT 1 FROM progress_bars
                WHERE progress_bars.id = trace_step_durations.progress_bar_id
                AND progress_bars.uid = ?
                AND progress_bars.version = ?
            )
            AND trace_step_durations.step_position = ?
        """,
        (
            bar_info.uid,
            bar_info.version,
            step_position,
        ),
    )
    if not response.results:
        return None
    return (
        float(hmean(np.array(response.results).reshape(len(response.results)))),
        None,
    )


async def get_step_eta_iterated_best_fit_linear(
    itgs: Itgs, bar_info: BarInfo, step_info: List[BarStepInfo], step_position: int
) -> Optional[Tuple[float, Optional[float]]]:
    """
    gets the eta for the given step using the best fit linear regression for the given progress bar

    Args:
        itgs (Itgs): the integrations to use
        bar_info (BarInfo): the progress bar to get the step eta percentile for
        step_info (List[BarStepInfo]): the steps for the progress bar
        step_position (int): the position of the step to get the eta for

    Returns:
        Optional[Tuple[float, Optional[float]]]: the step eta best fit linear regression if the progress bar exists and has traces
    """
    conn = await itgs.conn()
    cursor = conn.cursor("none")
    response = await cursor.execute(
        """
        SELECT 
            progress_bar_trace_steps.iterations,
            progress_bar_trace_steps.finished_at - progress_bar_trace_steps.started_at AS duration
        FROM progress_bar_trace_steps
        WHERE
            EXISTS (
                SELECT 1 FROM progress_bar_traces
                WHERE progress_bar_traces.id = progress_bar_trace_steps.progress_bar_trace_id
                AND EXISTS (
                    SELECT 1 FROM progress_bars
                    WHERE progress_bars.id = progress_bar_traces.progress_bar_id
                    AND progress_bars.uid = ?
                    AND progress_bars.version = ?
                )
            )
            AND EXISTS (
                SELECT 1 FROM progress_bar_steps
                WHERE progress_bar_steps.id = progress_bar_trace_steps.progress_bar_step_id
                AND progress_bar_steps.position = ?
            )
        """,
        (
            bar_info.uid,
            bar_info.version,
            step_position,
        ),
    )
    if response.results is None or len(response.results) < 2:
        return None
    data = np.array(response.results)
    x = data[:, 0]
    y = data[:, 1]
    if np.all(x == x[0]):
        return float(np.average(y) / x[0]), 0.0
    poly: Polynomial = Polynomial.fit(x, y, 1).convert()
    result = poly.coef
    return float(result[0]), float(result[1])
