import os
import pickle
import time
from typing import List, Dict, Any, Optional, Callable
from .common import (
    JobState, TaskState, TaskType, Task, InputSplit,
    WorkerStatus, generate_id
)


class Job:
    def __init__(
        self,
        job_id: str,
        name: str,
        input_data: List[Any],
        map_func: Callable,
        reduce_func: Callable,
        num_map_tasks: int = 0,
        num_reduce_tasks: int = 2,
        output_dir: str = "./output"
    ):
        self.job_id = job_id
        self.name = name
        self.input_data = input_data
        self.map_func = map_func
        self.reduce_func = reduce_func
        self.num_reduce_tasks = num_reduce_tasks
        self.output_dir = os.path.join(output_dir, job_id)
        self.state = JobState.PENDING
        self.map_tasks: List[Task] = []
        self.reduce_tasks: List[Task] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.map_outputs: Dict[str, List[str]] = {}

        if num_map_tasks <= 0:
            self.num_map_tasks = max(1, len(input_data) // 100 + 1)
        else:
            self.num_map_tasks = num_map_tasks

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "map-outputs"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "reduce-outputs"), exist_ok=True)

    def create_input_splits(self) -> List[InputSplit]:
        splits = []
        data_len = len(self.input_data)
        split_size = max(1, data_len // self.num_map_tasks)

        for i in range(self.num_map_tasks):
            start = i * split_size
            end = start + split_size if i < self.num_map_tasks - 1 else data_len
            if start >= data_len:
                break
            split_data = self.input_data[start:end]
            splits.append(InputSplit(
                split_id=f"split-{i}",
                data=split_data,
                start_idx=start,
                length=len(split_data)
            ))

        return splits

    def create_map_tasks(self) -> List[Task]:
        splits = self.create_input_splits()
        self.map_tasks = []
        for i, split in enumerate(splits):
            task = Task(
                task_id=f"{self.job_id}-m-{i}",
                task_type=TaskType.MAP,
                job_id=self.job_id,
                input_split=split,
                partition_id=None
            )
            self.map_tasks.append(task)
        return self.map_tasks

    def create_reduce_tasks(self) -> List[Task]:
        self.reduce_tasks = []
        for i in range(self.num_reduce_tasks):
            task = Task(
                task_id=f"{self.job_id}-r-{i}",
                task_type=TaskType.REDUCE,
                job_id=self.job_id,
                partition_id=i
            )
            self.reduce_tasks.append(task)
        return self.reduce_tasks

    @property
    def completed_map_tasks(self) -> int:
        return sum(1 for t in self.map_tasks if t.state == TaskState.COMPLETED)

    @property
    def completed_reduce_tasks(self) -> int:
        return sum(1 for t in self.reduce_tasks if t.state == TaskState.COMPLETED)

    @property
    def map_progress(self) -> float:
        if not self.map_tasks:
            return 0.0
        return self.completed_map_tasks / len(self.map_tasks)

    @property
    def reduce_progress(self) -> float:
        if not self.reduce_tasks:
            return 0.0
        return self.completed_reduce_tasks / len(self.reduce_tasks)

    def all_maps_completed(self) -> bool:
        return all(t.state == TaskState.COMPLETED for t in self.map_tasks)

    def all_reduces_completed(self) -> bool:
        return all(t.state == TaskState.COMPLETED for t in self.reduce_tasks)

    def any_task_failed(self) -> bool:
        return any(t.state == TaskState.FAILED and t.attempt >= 4 for t in self.map_tasks + self.reduce_tasks)


class JobTracker:
    def __init__(self):
        self.jobs: Dict[str, Job] = {}
        self.job_queue: List[str] = []
        self.completed_jobs: List[str] = []
        self.failed_jobs: List[str] = []

    def submit_job(
        self,
        name: str,
        input_data: List[Any],
        map_func: Callable,
        reduce_func: Callable,
        num_map_tasks: int = 0,
        num_reduce_tasks: int = 2,
        output_dir: str = "./output"
    ) -> str:
        job_id = generate_id()
        job = Job(
            job_id=job_id,
            name=name,
            input_data=input_data,
            map_func=map_func,
            reduce_func=reduce_func,
            num_map_tasks=num_map_tasks,
            num_reduce_tasks=num_reduce_tasks,
            output_dir=output_dir
        )
        job.create_map_tasks()
        job.create_reduce_tasks()
        self.jobs[job_id] = job
        self.job_queue.append(job_id)
        return job_id

    def get_job(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        job = self.get_job(job_id)
        if not job:
            return {"error": "Job not found"}
        return {
            "job_id": job.job_id,
            "name": job.name,
            "state": job.state.value,
            "map_progress": job.map_progress,
            "reduce_progress": job.reduce_progress,
            "num_map_tasks": len(job.map_tasks),
            "num_reduce_tasks": len(job.reduce_tasks),
            "completed_maps": job.completed_map_tasks,
            "completed_reduces": job.completed_reduce_tasks,
            "start_time": job.start_time,
            "end_time": job.end_time
        }

    def update_job_state(self, job_id: str, new_state: JobState):
        job = self.get_job(job_id)
        if job:
            job.state = new_state
            if new_state == JobState.RUNNING and job.start_time is None:
                job.start_time = time.time()
            if new_state in (JobState.SUCCEEDED, JobState.FAILED):
                job.end_time = time.time()
                if new_state == JobState.SUCCEEDED:
                    if job_id not in self.completed_jobs:
                        self.completed_jobs.append(job_id)
                else:
                    if job_id not in self.failed_jobs:
                        self.failed_jobs.append(job_id)

    def list_jobs(self) -> List[Dict[str, Any]]:
        return [self.get_job_status(jid) for jid in self.jobs]

    def get_pending_tasks(self, job_id: str, task_type: TaskType) -> List[Task]:
        job = self.get_job(job_id)
        if not job:
            return []
        tasks = job.map_tasks if task_type == TaskType.MAP else job.reduce_tasks
        return [t for t in tasks if t.state == TaskState.PENDING]

    def get_running_tasks(self, job_id: str, task_type: Optional[TaskType] = None) -> List[Task]:
        job = self.get_job(job_id)
        if not job:
            return []
        if task_type == TaskType.MAP:
            return [t for t in job.map_tasks if t.state in (TaskState.ASSIGNED, TaskState.RUNNING)]
        elif task_type == TaskType.REDUCE:
            return [t for t in job.reduce_tasks if t.state in (TaskState.ASSIGNED, TaskState.RUNNING)]
        else:
            all_tasks = job.map_tasks + job.reduce_tasks
            return [t for t in all_tasks if t.state in (TaskState.ASSIGNED, TaskState.RUNNING)]

    def get_task_by_id(self, job_id: str, task_id: str) -> Optional[Task]:
        job = self.get_job(job_id)
        if not job:
            return None
        for t in job.map_tasks + job.reduce_tasks:
            if t.task_id == task_id:
                return t
        return None
