import os
import time
from typing import Optional, Any, List

import redis


class RedisWrapper:

    def __init__(
            self,
            host: str = 'localhost',
            port: int = 6379,
            db: int = 0,
            password: Optional[str] = None
    ):
        redis_password = password if password is not None else os.getenv('REDIS_PASSWORD')
        self.host: str = host
        self.port: int = port
        self.db: int = db
        self.redis_connection: redis.Redis = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=redis_password
        )

    def save(self, key: str, value: str) -> bool:
        result = self.redis_connection.set(key, value.encode('utf-8'))
        if result:
            return True
        else:
            return False

    def save_list(self, key: str, values: list) -> bool:
        result = self.redis_connection.rpush(key, *values)
        if result == len(values):
            return True
        else:
            return False

    def update(self, key: str, value):
        return self.save(key, value)

    def append_to_list(self, key: str, values):
        return self.save_list(key, values)

    def get(self, key: str) -> Optional[Any]:
        value = self.redis_connection.get(key)
        if value is not None:
            result = value.decode('utf-8')
            return result
        else:
            return None

    def get_list(self, key: str) -> Optional[List[Any]]:
        values = self.redis_connection.lrange(key, 0, -1)
        if len(values) > 0:
            result = [v.decode('utf-8') for v in values]
            return result
        else:
            return None

    def get_all(self, key_pattern: Optional[str] = None) -> Optional[List[Any]]:
        pattern = key_pattern if key_pattern is not None else '*'
        keys = self.redis_connection.keys(pattern)
        results_str = self.redis_connection.mget(keys)
        results_str = [i.decode('utf-8') for i in results_str]
        return results_str

    def get_keys(self, key_pattern: Optional[str] = None) -> List[str]:
        pattern = key_pattern if key_pattern is not None else '*'
        keys = self.redis_connection.keys(pattern)
        results_str = [i.decode('utf-8') for i in keys]
        return results_str

    def delete(self, key: str) -> Optional[int]:
        return self.redis_connection.delete(key)

    def delete_all(self, pattern: str):
        keys = self.redis_connection.keys(pattern)
        if len(keys) > 0:
            return self.redis_connection.delete(*keys)

    def check_connection(self) -> bool:
        try:
            return self.redis_connection.ping()
        except redis.ConnectionError:
            return False

    def close(self):
        self.redis_connection.close()
        self.redis_connection.client().close()
        self.redis_connection.client().connection_pool.disconnect()


class RedisQueue:

    def __init__(
            self,
            host: str = 'localhost',
            port: int = 6379,
            db: int = 0,
            password: Optional[str] = None
    ):
        self.host: str = host
        self.port: int = port
        self.db: int = db
        self.redis_connection: redis.Redis = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password
        )
        self.redis_connection.config_set('notify-keyspace-events', 'KA')
        self.pub_sub = None

    def push(self, key, value, allow_duplicates=True):
        if allow_duplicates is True:
            return self.redis_connection.rpush(key, value)
        else:
            queue_elements = self.redis_connection.lrange(key, 0, -1)
            if value.encode('utf-8') in queue_elements:
                # No push is performed, because the task_uuid is already present in the queue
                return 0
            else:
                return self.redis_connection.rpush(key, value)

    def get(self, key, timeout=60, is_keyspace=False):
        if self.pub_sub is None:
            self.pub_sub = self.redis_connection.pubsub()
        subscribe_key = key
        if is_keyspace:
            subscribe_key = f'__keyspace@{self.db}__:{key}'
        self.pub_sub.subscribe(subscribe_key)
        data = None
        start = time.time()
        while data is None:
            message = self.pub_sub.get_message(ignore_subscribe_messages=True)
            if message is not None and 'type' in message and message['type'] == 'message':
                data = message['data']
            now = time.time()
            if now - start > timeout:
                raise TimeoutError('No message received on key {} in the latest {} seconds'.format(key, timeout))
            time.sleep(0.001)
        return data

    def subscribe(self, key, callback, db=None):
        if db is None:
            db = self.db

        def _internal_callback(message):
            data = message['data'].decode('utf-8')
            if data == 'rpush':
                item = self.redis_connection.lpop(key)
                if item is not None:
                    callback(item.decode('utf-8'))
                else:
                    callback(item)

        pubsub = self.redis_connection.pubsub()
        pubsub.subscribe(**{f'__keyspace@{db}__:{key}': _internal_callback})
        return pubsub.run_in_thread(sleep_time=0.001)

    def close(self):
        if self.pub_sub is not None:
            self.pub_sub.close()
            self.pub_sub.connection_pool.disconnect()
            time.sleep(0.01)
        self.redis_connection.close()
        self.redis_connection.client().close()
        self.redis_connection.client().connection_pool.disconnect()
