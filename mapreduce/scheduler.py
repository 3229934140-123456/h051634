import time
from typing import List, Dict, Optional, Tuple
from .common import (
    JobState, TaskState, TaskType, Task,
    WorkerStatus, generate_id
)
from .job import JobTracker, Job


class TaskScheduler:
    def __init__(self, job_tracker: JobTracker, workers: Dict[str, WorkerStatus]):
        self.job_tracker = job_tracker
        self.workers = workers
        self.speculative_threshold = 2.0
        self.speculative_check_interval = 5.0
        self.last_speculative_check = 0.0

    def assign_next_task(self, worker_id: str) -> Optional[Task]:
        worker = self.workers.get(worker_id)
        if not worker or not worker.is_alive or not worker.available:
            return None

        task = self._assign_map_task(worker)
        if task:
            return task

        task = self._assign_reduce_task(worker)
        if task:
            return task

        return None

    def _assign_map_task(self, worker: WorkerStatus) -> Optional[Task]:
        running_maps = len([t for t in worker.running_tasks if "-m-" in t])
        if running_maps >= worker.num_map_slots:
            return None

        for job_id in self.job_tracker.job_queue:
            job = self.job_tracker.get_job(job_id)
            if not job or job.state not in (JobState.PENDING, JobState.RUNNING):
                continue

            pending_maps = [t for t in job.map_tasks if t.state == TaskState.PENDING]
            if pending_maps:
                task = pending_maps[0]
                self._assign_task(task, worker)
                if job.state == JobState.PENDING:
                    self.job_tracker.update_job_state(job_id, JobState.RUNNING)
                return task

        return None

    def _assign_reduce_task(self, worker: WorkerStatus) -> Optional[Task]:
        running_reduces = len([t for t in worker.running_tasks if "-r-" in t])
        if running_reduces >= worker.num_reduce_slots:
            return None

        for job_id in self.job_tracker.job_queue:
            job = self.job_tracker.get_job(job_id)
            if not job:
                continue
            if job.state not in (JobState.RUNNING, JobState.MAP_COMPLETED):
                continue
            if not job.all_maps_completed():
                continue

            pending_reduces = [t for t in job.reduce_tasks if t.state == TaskState.PENDING]
            if pending_reduces:
                task = pending_reduces[0]
                self._assign_task(task, worker)
                if job.state == JobState.RUNNING:
                    self.job_tracker.update_job_state(job_id, JobState.MAP_COMPLETED)
                return task

        return None

    def _assign_task(self, task: Task, worker: WorkerStatus):
        task.state = TaskState.ASSIGNED
        task.worker_id = worker.worker_id
        task.attempt += 1
        worker.running_tasks.append(task.task_id)
        worker.available = len(worker.running_tasks) < (worker.num_map_slots + worker.num_reduce_slots)

    def mark_task_running(self, job_id: str, task_id: str):
        task = self.job_tracker.get_task_by_id(job_id, task_id)
        if task and task.state == TaskState.ASSIGNED:
            task.mark_running(task.worker_id or "")

    def complete_task(self, job_id: str, task_id: str, output_path: str) -> bool:
        task = self.job_tracker.get_task_by_id(job_id, task_id)
        if not task:
            return False

        task.mark_completed(output_path)

        worker = self.workers.get(task.worker_id) if task.worker_id else None
        if worker and task_id in worker.running_tasks:
            worker.running_tasks.remove(task_id)
            worker.available = len(worker.running_tasks) < (worker.num_map_slots + worker.num_reduce_slots)

        job = self.job_tracker.get_job(job_id)
        if job:
            if task.task_type == TaskType.MAP:
                if task.partition_id is not None:
                    if task.worker_id not in job.map_outputs:
                        job.map_outputs[task.worker_id] = []
                    job.map_outputs[task.worker_id].append(output_path)

            if job.all_maps_completed() and job.state == JobState.RUNNING:
                self.job_tracker.update_job_state(job_id, JobState.MAP_COMPLETED)

            if job.all_reduces_completed():
                self.job_tracker.update_job_state(job_id, JobState.REDUCE_COMPLETED)
                self.job_tracker.update_job_state(job_id, JobState.SUCCEEDED)

        return True

    def fail_task(self, job_id: str, task_id: str) -> bool:
        task = self.job_tracker.get_task_by_id(job_id, task_id)
        if not task:
            return False

        task.mark_failed()

        worker = self.workers.get(task.worker_id) if task.worker_id else None
        if worker and task_id in worker.running_tasks:
            worker.running_tasks.remove(task_id)
            worker.available = len(worker.running_tasks) < (worker.num_map_slots + worker.num_reduce_slots)

        if task.attempt < 4:
            task.state = TaskState.PENDING
            task.worker_id = None
            task.start_time = None
            task.end_time = None
            task.is_speculative = False
        else:
            job = self.job_tracker.get_job(job_id)
            if job:
                self.job_tracker.update_job_state(job_id, JobState.FAILED)

        return True

    def check_speculative_execution(self):
        now = time.time()
        if now - self.last_speculative_check < self.speculative_check_interval:
            return
        self.last_speculative_check = now

        for job_id in self.job_tracker.job_queue:
            job = self.job_tracker.get_job(job_id)
            if not job or job.state not in (JobState.RUNNING, JobState.MAP_COMPLETED):
                continue

            self._check_speculative_for_tasks(job, job.map_tasks)
            self._check_speculative_for_tasks(job, job.reduce_tasks)

    def _check_speculative_for_tasks(self, job: Job, tasks: List[Task]):
        completed = [t for t in tasks if t.state == TaskState.COMPLETED and not t.is_speculative]
        if len(completed) < max(1, len(tasks) // 2):
            return

        avg_duration = sum(t.duration for t in completed) / len(completed)

        running = [t for t in tasks if t.state == TaskState.RUNNING and not t.is_speculative]
        for task in running:
            if task.start_time and (time.time() - task.start_time) > avg_duration * self.speculative_threshold:
                if task.attempt < 3:
                    self._launch_speculative_copy(job, task)

    def _launch_speculative_copy(self, job: Job, original_task: Task):
        for t in (job.map_tasks + job.reduce_tasks):
            if t.task_id == original_task.task_id and t.is_speculative:
                return

        spec_task_id = f"{original_task.task_id}-spec-{generate_id()[:4]}"
        spec_task = Task(
            task_id=spec_task_id,
            task_type=original_task.task_type,
            job_id=original_task.job_id,
            state=TaskState.PENDING,
            attempt=original_task.attempt + 1,
            input_split=original_task.input_split,
            partition_id=original_task.partition_id,
            is_speculative=True
        )

        if original_task.task_type == TaskType.MAP:
            job.map_tasks.append(spec_task)
        else:
            job.reduce_tasks.append(spec_task)

    def handle_worker_failure(self, worker_id: str):
        worker = self.workers.get(worker_id)
        if not worker:
            return

        failed_task_ids = list(worker.running_tasks)
        worker.running_tasks.clear()
        worker.available = False

        for job_id in self.job_tracker.job_queue:
            job = self.job_tracker.get_job(job_id)
            if not job:
                continue

            all_tasks = job.map_tasks + job.reduce_tasks
            for task in all_tasks:
                if task.task_id in failed_task_ids:
                    if task.state in (TaskState.ASSIGNED, TaskState.RUNNING):
                        self.fail_task(job_id, task.task_id)

            if worker_id in job.map_outputs:
                del job.map_outputs[worker_id]
                for map_task in job.map_tasks:
                    if map_task.worker_id == worker_id and map_task.state == TaskState.COMPLETED:
                        map_task.state = TaskState.PENDING
                        map_task.worker_id = None
                        map_task.output_path = None
                        map_task.start_time = None
                        map_task.end_time = None

                if job.state in (JobState.MAP_COMPLETED, JobState.REDUCE_COMPLETED):
                    if not job.all_maps_completed():
                        self.job_tracker.update_job_state(job_id, JobState.RUNNING)

    def check_slow_workers(self) -> List[str]:
        slow_workers = []
        for worker_id, worker in self.workers.items():
            if not worker.is_alive:
                slow_workers.append(worker_id)
        return slow_workers
