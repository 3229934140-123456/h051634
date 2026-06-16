import time
import threading
from typing import Dict, List, Optional, Callable, Any, Tuple
from .common import (
    TaskState, TaskType, Task, WorkerStatus,
    generate_id
)
from .job import Job
from .shuffle import ShuffleManager, group_by_key


class Worker:
    def __init__(
        self,
        worker_id: str,
        host: str = "localhost",
        port: int = 0,
        num_map_slots: int = 2,
        num_reduce_slots: int = 2,
        shuffle_manager: Optional[ShuffleManager] = None
    ):
        self.worker_id = worker_id
        self.host = host
        self.port = port
        self.status = WorkerStatus(
            worker_id=worker_id,
            host=host,
            port=port,
            num_map_slots=num_map_slots,
            num_reduce_slots=num_reduce_slots
        )
        self.shuffle_manager = shuffle_manager
        self.running: bool = False
        self._task_threads: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()
        self._jobs: Dict[str, Job] = {}

    def start(self):
        self.running = True
        self.status.available = True

    def stop(self):
        self.running = False
        for thread in self._task_threads.values():
            thread.join(timeout=1.0)

    def heartbeat(self) -> WorkerStatus:
        self.status.last_heartbeat = time.time()
        return self.status

    def register_job(self, job: Job):
        self._jobs[job.job_id] = job

    def can_accept_task(self, task_type: TaskType) -> bool:
        with self._lock:
            running_maps = len([t for t in self.status.running_tasks if "-m-" in t])
            running_reduces = len([t for t in self.status.running_tasks if "-r-" in t])

            if task_type == TaskType.MAP:
                return running_maps < self.status.num_map_slots
            else:
                return running_reduces < self.status.num_reduce_slots

    def assign_task(self, task: Task, job: Job) -> bool:
        if not self.can_accept_task(task.task_type):
            return False

        with self._lock:
            self.status.running_tasks.append(task.task_id)
            self._jobs[job.job_id] = job

        thread = threading.Thread(
            target=self._run_task,
            args=(task, job),
            daemon=True
        )
        self._task_threads[task.task_id] = thread
        thread.start()
        return True

    def _run_task(self, task: Task, job: Job):
        try:
            task.mark_running(self.worker_id)

            if task.task_type == TaskType.MAP:
                output_path = self._execute_map_task(task, job)
            else:
                output_path = self._execute_reduce_task(task, job)

            task.mark_completed(output_path)
            self._on_task_complete(task, job, True, output_path)

        except Exception as e:
            print(f"[Worker {self.worker_id}] Task {task.task_id} failed: {e}")
            task.mark_failed()
            self._on_task_complete(task, job, False, None)

    def _execute_map_task(self, task: Task, job: Job) -> str:
        if not task.input_split:
            raise ValueError("Map task has no input split")

        map_output: List[Tuple[Any, Any]] = []
        for item in task.input_split.data:
            result = job.map_func(item)
            if isinstance(result, list):
                map_output.extend(result)
            else:
                map_output.append(result)

        if self.shuffle_manager:
            output_files = self.shuffle_manager.process_map_output(
                job.job_id,
                task.task_id,
                map_output,
                job.num_reduce_tasks
            )
            return list(output_files.values())[0] if output_files else ""
        else:
            return ""

    def _execute_reduce_task(self, task: Task, job: Job) -> str:
        if task.partition_id is None:
            raise ValueError("Reduce task has no partition id")

        accepted_map_ids = [t.task_id for t in job.map_tasks if t.result_accepted]

        if self.shuffle_manager:
            shuffled_data = self.shuffle_manager.get_partition_inputs(
                job.job_id,
                task.partition_id,
                accepted_map_ids
            )
        else:
            shuffled_data = []

        sorted_data = sorted(shuffled_data, key=lambda x: str(x[0]))
        grouped = group_by_key(sorted_data)

        results: List[Tuple[Any, Any]] = []
        for key, values in grouped:
            result = job.reduce_func(key, values)
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], tuple) and len(result[0]) == 2:
                results.extend(result)
            else:
                results.append((key, result))

        if self.shuffle_manager:
            return self.shuffle_manager.write_reduce_output(
                job.job_id,
                task.task_id,
                results
            )
        else:
            return ""

    def _on_task_complete(self, task: Task, job: Job, success: bool, output_path: Optional[str]):
        with self._lock:
            if task.task_id in self.status.running_tasks:
                self.status.running_tasks.remove(task.task_id)
            self.status.available = len(self.status.running_tasks) < (
                self.status.num_map_slots + self.status.num_reduce_slots
            )
            if task.task_id in self._task_threads:
                del self._task_threads[task.task_id]

    def get_running_task_count(self) -> int:
        with self._lock:
            return len(self.status.running_tasks)

    def has_capacity(self) -> bool:
        with self._lock:
            return len(self.status.running_tasks) < (
                self.status.num_map_slots + self.status.num_reduce_slots
            )
