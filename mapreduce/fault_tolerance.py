import time
import threading
from typing import Dict, List, Optional, Set
from .common import WorkerStatus, TaskState, JobState, TaskType
from .job import JobTracker, Job
from .scheduler import TaskScheduler


class FaultToleranceManager:
    def __init__(
        self,
        job_tracker: JobTracker,
        scheduler: TaskScheduler,
        workers: Dict[str, WorkerStatus]
    ):
        self.job_tracker = job_tracker
        self.scheduler = scheduler
        self.workers = workers
        self.failed_workers: Set[str] = set()
        self.heartbeat_timeout = 30.0
        self.max_task_attempts = 4
        self._monitor_thread: Optional[threading.Thread] = None
        self._running = False

    def start_monitor(self):
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop_monitor(self):
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

    def _monitor_loop(self):
        while self._running:
            self.check_worker_heartbeats()
            self.check_task_timeouts()
            self.scheduler.check_speculative_execution()
            time.sleep(2.0)

    def check_worker_heartbeats(self):
        now = time.time()
        dead_workers = []

        for worker_id, worker in self.workers.items():
            if not worker.is_alive and worker_id not in self.failed_workers:
                dead_workers.append(worker_id)

        for worker_id in dead_workers:
            self._handle_worker_death(worker_id)

    def _handle_worker_death(self, worker_id: str):
        print(f"[FaultTolerance] Worker {worker_id} detected as dead")
        self.failed_workers.add(worker_id)
        self.scheduler.handle_worker_failure(worker_id)

    def check_task_timeouts(self):
        now = time.time()
        timeout_threshold = 300.0

        for job_id in self.job_tracker.job_queue:
            job = self.job_tracker.get_job(job_id)
            if not job or job.state in (JobState.SUCCEEDED, JobState.FAILED):
                continue

            all_tasks = job.map_tasks + job.reduce_tasks
            for task in all_tasks:
                if task.state == TaskState.RUNNING and task.start_time:
                    elapsed = now - task.start_time
                    if elapsed > timeout_threshold and not task.is_speculative:
                        print(f"[FaultTolerance] Task {task.task_id} timed out after {elapsed:.1f}s")
                        self.scheduler.fail_task(job_id, task.task_id)

    def worker_heartbeat(self, worker_id: str) -> bool:
        worker = self.workers.get(worker_id)
        if not worker:
            return False
        worker.last_heartbeat = time.time()
        return True

    def get_failed_workers(self) -> List[str]:
        return list(self.failed_workers)

    def can_worker_rejoin(self, worker_id: str) -> bool:
        return worker_id in self.failed_workers

    def rejoin_worker(self, worker: WorkerStatus):
        if worker.worker_id in self.failed_workers:
            self.failed_workers.remove(worker.worker_id)
        worker.last_heartbeat = time.time()
        worker.available = True
        self.workers[worker.worker_id] = worker

    def get_fault_tolerance_stats(self) -> Dict:
        stats = {
            "total_workers": len(self.workers),
            "alive_workers": sum(1 for w in self.workers.values() if w.is_alive),
            "dead_workers": len(self.failed_workers),
            "failed_workers": list(self.failed_workers),
            "pending_retry_jobs": []
        }

        for job_id in self.job_tracker.job_queue:
            job = self.job_tracker.get_job(job_id)
            if job:
                failed_tasks = [
                    t.task_id for t in job.map_tasks + job.reduce_tasks
                    if t.state == TaskState.FAILED
                ]
                if failed_tasks:
                    stats["pending_retry_jobs"].append({
                        "job_id": job_id,
                        "failed_tasks": failed_tasks
                    })

        return stats
