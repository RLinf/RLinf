import logging
from typing import Awaitable
from time import time

import redis
import socket

logger = logging.getLogger(__name__)


class RedisQueue:

    class QueueOp:
        """Queue operations using list in Redis"""

        def __init__(self, rq: 'RedisQueue'):
            self.rq = rq

        def _peak_sync(self, queue_name):
            result = self.rq.redis.lrange(queue_name, 0, 0)
            if result:
                return result[0]
            return None

        async def _peak_async(self, queue_name):
            result = await self.rq.redis.lrange(queue_name, 0, 0)
            if result:
                return result[0]
            return None

        def peak(self, queue_name) -> bytes | Awaitable[bytes]:
            if self.rq.is_async:
                return self._peak_async(queue_name)
            else:
                return self._peak_sync(queue_name)

        def push(self, queue_name, *values):
            return self.rq.redis.rpush(queue_name, *values)

        def pop(self, queue_name):
            return self.rq.redis.lpop(queue_name)

        def pop_multi(self, *queue_names):
            if not queue_names:
                return []
            pp = self.rq.redis.pipeline(transaction=False)
            for queue_name in queue_names:
                pp.lpop(queue_name)
            return pp.execute()

        def _block_pop_sync(self, *queue_names, timeout=0) -> tuple[str, bytes] | None:
            start = time()
            while True:
                effective_timeout = self.rq._get_proper_timeout(start, timeout)
                if effective_timeout <= 0:
                    break
                result = self.rq.redis.blpop(queue_names, timeout=effective_timeout)
                if result:
                    return result
            return None

        async def _block_pop_async(self, *queue_names, timeout=0) -> tuple[str, bytes] | None:
            start = time()
            while True:
                effective_timeout = self.rq._get_proper_timeout(start, timeout)
                if effective_timeout <= 0:
                    break
                result = await self.rq.redis.blpop(queue_names, timeout=effective_timeout)
                if result:
                    return result
            return None

        def block_pop(self, *queue_names, timeout=0) -> tuple[str, bytes] | None | Awaitable[tuple[str, bytes]] | Awaitable[None]:
            assert queue_names and all(isinstance(q, str) for q in queue_names), "queue_names must be a non-empty list of strings"
            if self.rq.is_async:
                return self._block_pop_async(*queue_names, timeout=timeout)
            else:
                return self._block_pop_sync(*queue_names, timeout=timeout)

        def len(self, queue_name):
            return self.rq.redis.llen(queue_name)

    class PriorityQueueOp:
        """Priority Queue operations using sorted set in Redis"""

        def __init__(self, rq: 'RedisQueue'):
            self.rq = rq

        def _peak_sync(self, queue_name):
            result = self.rq.redis.zrange(queue_name, 0, 0, withscores=True)
            if result:
                return result[0]
            return None

        async def _peak_async(self, queue_name):
            result = await self.rq.redis.zrange(queue_name, 0, 0, withscores=True)
            if result:
                return result[0]
            return None

        def peak(self, queue_name) -> tuple[bytes, float] | Awaitable[tuple[bytes, float]] | None | Awaitable[None]:
            if self.rq.is_async:
                return self._peak_async(queue_name)
            else:
                return self._peak_sync(queue_name)

        def push(self, queue_name, key_score_dict: dict[str, float]):
            return self.rq.redis.zadd(queue_name, key_score_dict)

        def pop(self, queue_name)-> tuple[bytes, float] | Awaitable[tuple[bytes, float]] | None | Awaitable[None]:
            return self.rq.redis.zpopmin(queue_name)

        def pop_multi(self, *queue_names):
            if not queue_names:
                return []
            pp = self.rq.redis.pipeline(transaction=False)
            for queue_name in queue_names:
                pp.zpopmin(queue_name)
            return pp.execute()

        def _block_pop_sync(self, *queue_names, timeout=0) -> tuple[str, bytes, float] | None:
            start = time()
            while True:
                effective_timeout = self.rq._get_proper_timeout(start, timeout)
                if effective_timeout <= 0:
                    break
                result = self.rq.redis.bzpopmin(queue_names, timeout=effective_timeout)
                if result:
                    return result
            return None

        async def _block_pop_async(self, *queue_names, timeout=0) -> tuple[str, bytes, float] | None:
            start = time()
            while True:
                effective_timeout = self.rq._get_proper_timeout(start, timeout)
                if effective_timeout <= 0:
                    break
                result = await self.rq.redis.bzpopmin(queue_names, timeout=effective_timeout)
                if result:
                    return result
            return None

        def block_pop(self, *queue_names, timeout=0) -> tuple[str, bytes, float] | None | Awaitable[tuple[str, bytes, float]] | Awaitable[None]:
            assert queue_names and all(isinstance(q, str) for q in queue_names), "queue_names must be a non-empty list of strings"
            if self.rq.is_async:
                return self._block_pop_async(*queue_names, timeout=timeout)
            else:
                return self._block_pop_sync(*queue_names, timeout=timeout)

        def len(self, queue_name):
            return self.rq.redis.zcard(queue_name)

    def __init__(self, redis_uri, *, socket_timeout: int = None, is_async: bool = False):
        self.redis_uri = redis_uri
        self.is_async = is_async
        self.socket_timeout = socket_timeout
        if self.socket_timeout is not None and self.socket_timeout < 10:
            raise ValueError('socket_timeout must be at least 10 seconds')
        self.redis: redis.Redis | redis.asyncio.Redis = self._init_redis(socket_timeout)
        self.queue = self.QueueOp(self)
        self.pqueue = self.PriorityQueueOp(self)

    def _init_redis(self, socket_timeout) -> redis.Redis | redis.asyncio.Redis:
        if '+cluster://' in self.redis_uri:
            Redis = redis.RedisCluster if not self.is_async else redis.asyncio.RedisCluster
            redis_uri = self.redis_uri.replace('+cluster://', '://')
        else:
            Redis = redis.Redis if not self.is_async else redis.asyncio.Redis
            redis_uri = self.redis_uri

        return Redis.from_url(
            redis_uri,
            socket_connect_timeout=120,
            socket_timeout=socket_timeout,
            socket_keepalive=True,
            health_check_interval=30,
            socket_keepalive_options={socket.TCP_KEEPIDLE: 2, socket.TCP_KEEPINTVL: 1, socket.TCP_KEEPCNT: 2}
        )

    def ping(self):
        return self.redis.ping()

    def set(self, key, value, expire=None):
        return self.redis.set(key, value, ex=expire)

    def get(self, key):
        return self.redis.get(key)

    async def _count_keys_async(self, pattern):
        count = 0
        async for _ in self.redis.scan_iter(pattern, count=100):
            count += 1
        return count

    def _count_keys_sync(self, pattern):
        count = 0
        for _ in self.redis.scan_iter(pattern, count=100):
            count += 1
        return count

    def count_keys(self, pattern) -> int | Awaitable[int]:
        if self.is_async:
            return self._count_keys_async(pattern)
        else:
            return self._count_keys_sync(pattern)

    def expire(self, key, timeout):
        return self.redis.expire(key, timeout)

    def delete(self, *keys):
        return self.redis.delete(*keys)

    def _time_sync(self) -> float:
        t = self.redis.time()
        return t[0] + t[1] / 1_000_000

    async def _time_async(self) -> float:
        t = await self.redis.time()
        return t[0] + t[1] / 1_000_000

    def time(self) -> float | Awaitable[float]:
        if self.is_async:
            return self._time_async()
        else:
            return self._time_sync()

    def _get_proper_timeout(self, start, timeout):
        if timeout > 0:
            effective_timeout = timeout - int(time() - start)
        else:
            effective_timeout = self.socket_timeout
        effective_timeout = min(effective_timeout, self.socket_timeout - 2)  # 2 seconds for communication overhead
        # can be negative if timeout is too small
        return effective_timeout
