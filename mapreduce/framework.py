import time
import threading
import os
from typing import Dict, List, Optional, Callable, Any
from .common import (
    JobState, TaskState, TaskType, Task,
    WorkerStatus, generate_id
)
from .job import JobTracker, Job, JobReport
from .scheduler import TaskScheduler
from .worker import Worker
from .shuffle import ShuffleManager, read_input_files
from .fault_tolerance import FaultToleranceManager


class MapReduceFramework:
    def __init__(self, output_dir: str = "./output", num_workers: int = 3):
        self.output_dir = output_dir
        self.job_tracker = JobTracker()
        self.shuffle_manager = ShuffleManager(output_dir)
        self.workers: Dict[str, Worker] = {}
        self.worker_statuses: Dict[str, WorkerStatus] = {}
        self.scheduler = TaskScheduler(self.job_tracker, self.worker_statuses)
        self.fault_tolerance = FaultToleranceManager(
            self.job_tracker, self.scheduler, self.worker_statuses
        )
        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._status_monitor_thread: Optional[threading.Thread] = None
        self._status_callbacks: Dict[str, Callable] = {}

        os.makedirs(output_dir, exist_ok=True)

        for i in range(num_workers):
            self.add_worker(f"worker-{i}")

    def add_worker(self, worker_id: str, num_map_slots: int = 2, num_reduce_slots: int = 2) -> str:
        worker = Worker(
            worker_id=worker_id,
            num_map_slots=num_map_slots,
            num_reduce_slots=num_reduce_slots,
            shuffle_manager=self.shuffle_manager
        )
        self.workers[worker_id] = worker
        self.worker_statuses[worker_id] = worker.status
        return worker_id

    def start(self, monitor_status: bool = True):
        if self._running:
            return

        self._running = True
        for worker in self.workers.values():
            worker.start()

        self.fault_tolerance.start_monitor()

        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()

        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

        if monitor_status:
            self._status_monitor_thread = threading.Thread(target=self._status_monitor_loop, daemon=True)
            self._status_monitor_thread.start()

    def stop(self):
        self._running = False
        self.fault_tolerance.stop_monitor()
        for worker in self.workers.values():
            worker.stop()
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=2.0)
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=2.0)
        if self._status_monitor_thread:
            self._status_monitor_thread.join(timeout=2.0)

    def _scheduler_loop(self):
        while self._running:
            self._schedule_tasks()
            time.sleep(0.1)

    def _schedule_tasks(self):
        for worker_id, worker in self.workers.items():
            if not worker.status.is_alive or not worker.has_capacity():
                continue

            job = None
            for job_id in self.job_tracker.job_queue:
                j = self.job_tracker.get_job(job_id)
                if j and j.state not in (JobState.SUCCEEDED, JobState.FAILED):
                    job = j
                    break

            if not job:
                continue

            for task_type in [TaskType.MAP, TaskType.REDUCE]:
                if not worker.can_accept_task(task_type):
                    continue

                if task_type == TaskType.REDUCE:
                    if not job.all_maps_completed():
                        continue

                pending_tasks = self.job_tracker.get_pending_tasks(job.job_id, task_type)
                if pending_tasks:
                    task = pending_tasks[0]
                    self._assign_task_to_worker(task, worker, job)
                    break

    def _assign_task_to_worker(self, task: Task, worker: Worker, job: Job):
        task.state = TaskState.ASSIGNED
        task.worker_id = worker.worker_id
        task.attempt += 1
        worker.status.running_tasks.append(task.task_id)
        worker.register_job(job)

        if job.state == JobState.PENDING:
            self.job_tracker.update_job_state(job.job_id, JobState.RUNNING)
            self._notify_status_change(job.job_id)

        thread = threading.Thread(
            target=self._execute_task,
            args=(task, worker, job),
            daemon=True
        )
        thread.start()

    def _execute_task(self, task: Task, worker: Worker, job: Job):
        try:
            task.mark_running(worker.worker_id)
            self._notify_status_change(job.job_id)

            if task.task_type == TaskType.MAP:
                output_path = self._run_map_task(task, job)
            else:
                output_path = self._run_reduce_task(task, job)

            self.scheduler.complete_task(job.job_id, task.task_id, output_path)

            self._cleanup_speculative_tasks(task, job)

        except Exception as e:
            print(f"[Framework] Task {task.task_id} failed: {e}")
            self.scheduler.fail_task(job.job_id, task.task_id)

        finally:
            if task.task_id in worker.status.running_tasks:
                worker.status.running_tasks.remove(task.task_id)
            worker.status.available = len(worker.status.running_tasks) < (
                worker.status.num_map_slots + worker.status.num_reduce_slots
            )
            self._notify_status_change(job.job_id)

    def _run_map_task(self, task: Task, job: Job) -> str:
        if not task.input_split:
            raise ValueError("Map task has no input split")

        map_output = []
        for item in task.input_split.data:
            result = job.map_func(item)
            if isinstance(result, list):
                map_output.extend(result)
            else:
                map_output.append(result)

        output_files = self.shuffle_manager.process_map_output(
            job.job_id, task.task_id, map_output, job.num_reduce_tasks
        )

        first_path = list(output_files.values())[0] if output_files else ""
        task.output_path = first_path
        return first_path

    def _run_reduce_task(self, task: Task, job: Job) -> str:
        if task.partition_id is None:
            raise ValueError("Reduce task has no partition id")

        accepted_map_ids = [t.task_id for t in job.map_tasks if t.result_accepted]
        shuffled_data = self.shuffle_manager.get_partition_inputs(
            job.job_id, task.partition_id, accepted_map_ids
        )

        sorted_data = sorted(shuffled_data, key=lambda x: str(x[0]))

        grouped = []
        current_key = None
        current_values = []
        for key, value in sorted_data:
            if key != current_key:
                if current_key is not None:
                    grouped.append((current_key, current_values))
                current_key = key
                current_values = [value]
            else:
                current_values.append(value)
        if current_key is not None:
            grouped.append((current_key, current_values))

        results = []
        for key, values in grouped:
            result = job.reduce_func(key, values)
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], tuple) and len(result[0]) == 2:
                results.extend(result)
            else:
                results.append((key, result))

        return self.shuffle_manager.write_reduce_output(
            job.job_id, task.task_id, results
        )

    def _cleanup_speculative_tasks(self, completed_task: Task, job: Job):
        if not completed_task.result_accepted:
            return

        all_tasks = job.map_tasks if completed_task.task_type == TaskType.MAP else job.reduce_tasks
        for task in all_tasks:
            if (task.logical_task_id == completed_task.logical_task_id
                    and task.task_id != completed_task.task_id
                    and task.state in (TaskState.PENDING, TaskState.ASSIGNED, TaskState.RUNNING)):
                task.state = TaskState.FAILED
                task.end_time = time.time()

    def _heartbeat_loop(self):
        while self._running:
            for worker_id, worker in self.workers.items():
                if worker.status.is_alive:
                    worker.heartbeat()
            time.sleep(1.0)

    def _status_monitor_loop(self):
        while self._running:
            for job_id in list(self._status_callbacks.keys()):
                status = self.get_job_status(job_id)
                if status.get("state") in ("SUCCEEDED", "FAILED"):
                    callback = self._status_callbacks.pop(job_id, None)
                    if callback:
                        try:
                            callback(status)
                        except Exception:
                            pass
            time.sleep(0.5)

    def _notify_status_change(self, job_id: str):
        pass

    def submit_job(
        self,
        name: str,
        input_data: List[Any],
        map_func: Callable,
        reduce_func: Callable,
        num_map_tasks: int = 0,
        num_reduce_tasks: int = 2,
        output_format: str = "text"
    ) -> str:
        job_id = self.job_tracker.submit_job(
            name=name,
            input_data=input_data,
            map_func=map_func,
            reduce_func=reduce_func,
            num_map_tasks=num_map_tasks,
            num_reduce_tasks=num_reduce_tasks,
            output_dir=self.output_dir,
            output_format=output_format
        )
        return job_id

    def submit_job_from_files(
        self,
        name: str,
        input_dir: str,
        map_func: Callable,
        reduce_func: Callable,
        split_by: str = "lines",
        chunk_size: int = 100,
        num_map_tasks: int = 0,
        num_reduce_tasks: int = 2,
        output_format: str = "text"
    ) -> str:
        input_data = read_input_files(input_dir, split_by, chunk_size)
        return self.submit_job(
            name=name,
            input_data=input_data,
            map_func=map_func,
            reduce_func=reduce_func,
            num_map_tasks=num_map_tasks,
            num_reduce_tasks=num_reduce_tasks,
            output_format=output_format
        )

    def get_job_status(self, job_id: str) -> Dict:
        return self.job_tracker.get_job_status(job_id)

    def wait_for_job(self, job_id: str, timeout: float = 300.0, show_progress: bool = False) -> Dict:
        start_time = time.time()
        last_state = None
        last_progress = (-1, -1)

        while self._running and time.time() - start_time < timeout:
            status = self.job_tracker.get_job_status(job_id)
            state = status.get("state")
            map_p = int(status.get("map_progress", 0) * 100)
            reduce_p = int(status.get("reduce_progress", 0) * 100)

            if show_progress:
                if state != last_state or (map_p, reduce_p) != last_progress:
                    last_state = state
                    last_progress = (map_p, reduce_p)
                    bar_map = "=" * (map_p // 5) + " " * (20 - map_p // 5)
                    bar_reduce = "=" * (reduce_p // 5) + " " * (20 - reduce_p // 5)
                    print(f"\r[{state}] Map: [{bar_map}] {map_p:3d}%  Reduce: [{bar_reduce}] {reduce_p:3d}%", end="")

            if state in ("SUCCEEDED", "FAILED"):
                if show_progress:
                    print()
                return status

            time.sleep(0.3)

        if show_progress:
            print()
        return self.job_tracker.get_job_status(job_id)

    def get_job_results(self, job_id: str) -> List:
        job = self.job_tracker.get_job(job_id)
        if not job:
            return []
        return self.shuffle_manager.get_final_results(job_id, job.num_reduce_tasks)

    def get_job_report(self, job_id: str) -> Optional[JobReport]:
        return self.job_tracker.get_job_report(job_id)

    def finalize_job_output(self, job_id: str) -> Optional[str]:
        job = self.job_tracker.get_job(job_id)
        if not job or job.state != JobState.SUCCEEDED:
            return None

        results = self.get_job_results(job_id)
        output_path = self.shuffle_manager.write_final_results(
            job_id, results, job.output_format
        )

        report = self.get_job_report(job_id)
        if report:
            report.save(job.output_dir)

        return output_path

    def list_jobs(self) -> List[Dict]:
        return self.job_tracker.list_jobs()

    def kill_job(self, job_id: str) -> bool:
        job = self.job_tracker.get_job(job_id)
        if not job:
            return False
        self.job_tracker.update_job_state(job_id, JobState.FAILED)
        return True

    def get_cluster_status(self) -> Dict:
        return {
            "num_workers": len(self.workers),
            "active_workers": sum(1 for w in self.workers.values() if w.status.is_alive),
            "total_map_slots": sum(w.status.num_map_slots for w in self.workers.values()),
            "total_reduce_slots": sum(w.status.num_reduce_slots for w in self.workers.values()),
            "running_tasks": sum(len(w.status.running_tasks) for w in self.workers.values()),
            "jobs": {
                "total": len(self.job_tracker.jobs),
                "running": sum(1 for j in self.job_tracker.jobs.values() if j.state == JobState.RUNNING),
                "completed": len(self.job_tracker.completed_jobs),
                "failed": len(self.job_tracker.failed_jobs)
            }
        }

    def print_job_report(self, job_id: str):
        report = self.get_job_report(job_id)
        if report:
            report.pretty_print()
