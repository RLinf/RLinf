import logging

from app.worker_manager import WorkerManager

logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

if __name__ == '__main__':
    work_manager = WorkerManager()
    work_manager.run()
