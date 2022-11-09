import os
import secrets
import time
from typing import Optional
from itgs import Itgs


async def get_example_user_token(itgs: Itgs) -> str:
    """gets or creates and gets a token for the example user that can be
    provided in the authorization header"""
    redis = await itgs.redis()
    token: Optional[bytes] = await redis.get("example_user_token")
    if token is not None:
        return str(token, "utf-8")
    new_token = "ep_ut_" + secrets.token_urlsafe(48)
    uid = "ep_ut_uid_" + secrets.token_urlsafe(16)
    now = time.time()
    conn = await itgs.conn()
    cursor = conn.cursor()
    await cursor.execute(
        """
        INSERT INTO user_tokens(
            user_id,
            uid,
            token,
            name,
            created_at,
            expires_at
        ) SELECT
            users.id,
            ?,
            ?,
            ?,
            ?,
            ?
        FROM users
        WHERE users.sub = ?""",
        (
            uid,
            new_token,
            "example",
            now,
            now + 86400,
            os.environ["EXAMPLE_USER_SUB"],
        ),
    )
    await redis.set("example_user_token", bytes(new_token, "utf-8"), ex=86300)
    return new_token
