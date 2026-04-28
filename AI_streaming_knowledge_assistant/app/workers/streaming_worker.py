import base64
import json
import logging

from app.core.redis import get_redis_client
from app.infrastructure.redis_stream import RedisStreamClient

logger = logging.getLogger(__name__)


def run_worker():
    redis_client = get_redis_client()

    stream = RedisStreamClient(
        redis_client=redis_client,
        stream_name="audio_stream",
        group_name="audio_group",
    )

    consumer_name = "worker-1"

    logger.info("Streaming worker started")

    while True:
        messages = stream.read_audio(consumer_name)

        if not messages:
            continue

        for _, msg_list in messages:
            for message_id, data in msg_list:

                try:
                    session_id = data["session_id"]
                    audio = base64.b64decode(data["audio"])
                    meta = json.loads(data.get("meta", "{}"))

 
                    result = {
                        "session_id": session_id,
                        "text": "transcribed text here"
                    }

                    stream.publish_result(session_id, result)
                    stream.ack(message_id)

                except Exception as e:
                    logger.error(f"worker error: {e}")