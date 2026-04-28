import asyncio


class EmotionBatcher:

    def __init__(self, service, batch_size=32, timeout=0.05):
        self.service = service
        self.batch_size = batch_size
        self.timeout = timeout
        self.queue = []
        self.lock = asyncio.Lock()

    async def predict(self, audio, sr, start, end):

        loop = asyncio.get_event_loop()
        future = loop.create_future()

        async with self.lock:
            self.queue.append((audio, sr, start, end, future))

            if len(self.queue) >= self.batch_size:
                asyncio.create_task(self._flush())

        return await future

    async def _flush(self):

        await asyncio.sleep(self.timeout)

        async with self.lock:
            batch = self.queue
            self.queue = []

        if not batch:
            return

        results = []

        for audio, sr, start, end, _ in batch:
            pred = self.service.predict_segment(audio, sr, start, end)
            results.append(pred)

        for (_, _, _, _, f), r in zip(batch, results):
            f.set_result(r)