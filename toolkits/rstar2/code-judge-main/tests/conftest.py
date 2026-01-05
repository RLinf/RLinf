import signal
import pytest
import fakeredis
import os
import sys
import subprocess
import multiprocessing
from time import sleep

REDIS_PORT = 6388

if os.environ.get('REDIS_URI') is None:
    os.environ['REDIS_URI'] = f'redis://localhost:{REDIS_PORT}/7'

os.environ['RUN_WORKERS'] = '0'
os.environ['MAX_WORKERS'] = '4'


@pytest.fixture(scope='session')
def test_client():
    """
    Create a test client for the FastAPI app.
    """
    from app.main import app
    from fastapi.testclient import TestClient
    from app.libs.utils import nothrow_killpg

    def _start_fake_redis():
        fakeredis.TcpFakeServer.allow_reuse_address = True
        server = fakeredis.TcpFakeServer(('localhost', REDIS_PORT))
        server.serve_forever()

    # blpop will make the server blocks, and hard to kill
    # so here we run it in a seperate process instead of thread
    # Note allow_reuse_address is set to True to avoid address already in use error
    redis_process = multiprocessing.Process(target=_start_fake_redis)
    redis_process.start()

    sleep(1)  # wait for the server to start

    def _start_workers():
        os.setsid()  # new process group
        # Start the workers
        from app.worker_manager import WorkerManager

        work_manager = WorkerManager()
        work_manager.run()

    workers = multiprocessing.Process(target=_start_workers)
    workers.start()
    sleep(1)  # wait for the workers to start

    try:
        with TestClient(app) as client:
            yield client
    finally:
        # TODO: dirty work
        # we firstly send SIGTERM to the process group
        # so coverage have a chance to finish
        # but it is not really killing the process group
        # so we need to kill the process group with SIGKILL
        nothrow_killpg(pgid=workers.pid, sig=signal.SIGTERM)
        workers.join(timeout=1)
        nothrow_killpg(pgid=workers.pid)

        redis_process.kill()
        redis_process.join()
