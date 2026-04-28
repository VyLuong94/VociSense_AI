import json
import logging
import base64
from typing import Dict, Any, List, Optional

import redis

logger = logging.getLogger(__name__)


class RedisStreamClient:
    """
    Infrastructure layer:
    - Redis Stream orchestration
    - NO connection init here (DI required)
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        stream_name: str,
        group_name: str,
    ):
        self.redis = redis_client
        self.stream_name = stream_name
        self.group_name = group_name

        self._ensure_group()

    # =========================
    # INIT STREAM GROUP
    # =========================
    def _ensure_group(self):
        try:
            self.redis.xgroup_create(
                name=self.stream_name,
                groupname=self.group_name,
                id="0",
                mkstream=True,
            )
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    # =========================
    # PRODUCER
    # =========================
    def push_audio(
        self,
        session_id: str,
        audio_bytes: bytes,
        metadata: Optional[Dict[str, Any]] = None,
        maxlen: int = 100000,
    ) -> Optional[str]:
        try:
            data = {
                "session_id": session_id,
                "audio": base64.b64encode(audio_bytes).decode(),
                "meta": json.dumps(metadata or {}),
            }

            return self.redis.xadd(
                self.stream_name,
                data,
                maxlen=maxlen,
                approximate=True,
            )

        except Exception as e:
            logger.error("push_audio failed", extra={
                "session_id": session_id,
                "error": str(e),
            })
            return None

    # =========================
    # CONSUMER
    # =========================
    def read_audio(
        self,
        consumer_name: str,
        count: int = 1,
        block: int = 1000,
    ) -> List:
        try:
            return self.redis.xreadgroup(
                groupname=self.group_name,
                consumername=consumer_name,
                streams={self.stream_name: ">"},
                count=count,
                block=block,
            ) or []
        except Exception as e:
            logger.error("read_audio failed", extra={"error": str(e)})
            return []

    # =========================
    # ACK
    # =========================
    def ack(self, message_id: str) -> bool:
        try:
            self.redis.xack(self.stream_name, self.group_name, message_id)
            return True
        except Exception:
            return False

    # =========================
    # RESULT PUB/SUB
    # =========================
    def publish_result(self, session_id: str, result: Dict[str, Any]) -> bool:
        try:
            self.redis.publish(
                f"result:{session_id}",
                json.dumps(result),
            )
            return True
        except Exception:
            return False

    def subscribe(self, session_id: str):
        pubsub = self.redis.pubsub()
        pubsub.subscribe(f"result:{session_id}")
        return pubsub