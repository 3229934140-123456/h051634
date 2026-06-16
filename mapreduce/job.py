import os
import pickle
import time
import json
from typing import List, Dict, Any, Optional, Callable, Set
from .common import (
    JobState, TaskState, TaskType, Task, InputSplit,
    WorkerStatus, generate_id
)


class JobReport:
    def __init__(self, job_id: str, name: str):
        self.job_id = job_id
        self.name = name
        self.map_task_reports: List[Dict[str, Any]] = []
        self.reduce_task_reports: List[Dict[str, Any]] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.final_state: Optional[str] = None
        self.final_output_path: Optional[str] = None
        self.total_duration: float = 0.0

    def add_task_report(self, task: Task):
        report = {
            "task_id": task.task_id,
            "logical_task_id": task.logical_task_id,
            "task_type": task.task_type.value,
            "state": task.state.value,
            "worker_id": task.worker_id,
            "attempt": task.attempt,
            "is_speculative": task.is_speculative,
            "result_accepted": task.result_accepted,
            "start_time": task.start_time,
            "end_time": task.end_time,
            "duration": task.duration,
            "output_path": task.output_path,
            "input_split": {
                "split_id": task.input_split.split_id,
                "start_idx": task.input_split.start_idx,
                "length": task.input_split.length,
                "num_records": len(task.input_split.data) if task.input_split else None
            } if task.input_split else None,
            "partition_id": task.partition_id
        }
        if task.task_type == TaskType.MAP:
            self.map_task_reports.append(report)
        else:
            self.reduce_task_reports.append(report)

    @property
    def summary(self) -> Dict[str, int]:
        return {
            "total_map_tasks": len(self.map_task_reports),
            "total_reduce_tasks": len(self.reduce_task_reports),
            "accepted_map_results": sum(1 for r in self.map_task_reports if r["result_accepted"]),
            "accepted_reduce_results": sum(1 for r in self.reduce_task_reports if r["result_accepted"]),
            "failed_attempts": sum(1 for r in self.map_task_reports + self.reduce_task_reports if r["state"] == "FAILED"),
            "speculative_attempts": sum(1 for r in self.map_task_reports + self.reduce_task_reports if r["is_speculative"])
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "final_state": self.final_state,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_duration": self.total_duration,
            "final_output_path": self.final_output_path,
            "map_tasks": self.map_task_reports,
            "reduce_tasks": self.reduce_task_reports,
            "summary": self.summary
        }

    def save(self, output_dir: str):
        report_path = os.path.join(output_dir, "job_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def pretty_print(self):
        print(f"\n{'='*60}")
        print(f"📊 作业运行报告: {self.name}")
        print(f"{'='*60}")
        print(f"作业ID: {self.job_id}")
        print(f"最终状态: {self.final_state}")
        print(f"总耗时: {self.total_duration:.2f} 秒")
        print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time)) if self.start_time else 'N/A'}")
        print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.end_time)) if self.end_time else 'N/A'}")
        print(f"最终输出: {self.final_output_path}")
        print(f"\n📋 概览:")
        print(f"  Map 任务尝试: {self.summary['total_map_tasks']} 次 (成功采纳: {self.summary['accepted_map_results']})")
        print(f"  Reduce 任务尝试: {self.summary['total_reduce_tasks']} 次 (成功采纳: {self.summary['accepted_reduce_results']})")
        print(f"  失败尝试: {self.summary['failed_attempts']} 次")
        print(f"  推测执行: {self.summary['speculative_attempts']} 次")

        print(f"\n📦 Map 任务详情:")
        print(f"{'任务ID':<30} {'逻辑ID':<25} {'Worker':<10} {'耗时(s)':<10} {'重试':<6} {'采纳':<6} {'推测':<6}")
        print(f"{'-'*95}")
        for r in self.map_task_reports:
            spec = "是" if r["is_speculative"] else "否"
            accepted = "✅" if r["result_accepted"] else "❌"
            split_info = f"{r['input_split']['start_idx']}-{r['input_split']['start_idx'] + r['input_split']['length']}" if r["input_split"] else "N/A"
            print(f"{r['task_id']:<30} {r['logical_task_id']:<25} {str(r['worker_id']):<10} {r['duration']:<10.2f} {r['attempt']:<6} {accepted:<6} {spec:<6}")
            if r["input_split"]:
                print(f"  ↳ 分片: {split_info}, 记录数: {r['input_split']['num_records']}, 输出: {r['output_path']}")

        print(f"\n🔧 Reduce 任务详情:")
        print(f"{'任务ID':<30} {'逻辑ID':<25} {'Worker':<10} {'耗时(s)':<10} {'重试':<6} {'采纳':<6} {'推测':<6}")
        print(f"{'-'*95}")
        for r in self.reduce_task_reports:
            spec = "是" if r["is_speculative"] else "否"
            accepted = "✅" if r["result_accepted"] else "❌"
            print(f"{r['task_id']:<30} {r['logical_task_id']:<25} {str(r['worker_id']):<10} {r['duration']:<10.2f} {r['attempt']:<6} {accepted:<6} {spec:<6}")
            print(f"  ↳ 分区: {r['partition_id']}, 输出: {r['output_path']}")
        print(f"{'='*60}\n")


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
        output_dir: str = "./output",
        output_format: str = "text"
    ):
        self.job_id = job_id
        self.name = name
        self.input_data = input_data
        self.map_func = map_func
        self.reduce_func = reduce_func
        self.num_reduce_tasks = num_reduce_tasks
        self.output_dir = os.path.join(output_dir, job_id)
        self.output_format = output_format
        self.state = JobState.PENDING
        self.map_tasks: List[Task] = []
        self.reduce_tasks: List[Task] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.map_outputs: Dict[str, List[str]] = {}
        self.accepted_logical_tasks: Set[str] = set()
        self.report = JobReport(job_id, name)

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
            logical_id = f"{self.job_id}-m-{i}"
            task = Task(
                task_id=logical_id,
                task_type=TaskType.MAP,
                job_id=self.job_id,
                logical_task_id=logical_id,
                input_split=split,
                partition_id=None
            )
            self.map_tasks.append(task)
        return self.map_tasks

    def create_reduce_tasks(self) -> List[Task]:
        self.reduce_tasks = []
        for i in range(self.num_reduce_tasks):
            logical_id = f"{self.job_id}-r-{i}"
            task = Task(
                task_id=logical_id,
                task_type=TaskType.REDUCE,
                job_id=self.job_id,
                logical_task_id=logical_id,
                partition_id=i
            )
            self.reduce_tasks.append(task)
        return self.reduce_tasks

    def is_logical_task_completed(self, logical_task_id: str) -> bool:
        return logical_task_id in self.accepted_logical_tasks

    def accept_task_result(self, task: Task) -> bool:
        if task.logical_task_id in self.accepted_logical_tasks:
            return False
        self.accepted_logical_tasks.add(task.logical_task_id)
        task.result_accepted = True
        return True

    @property
    def completed_map_tasks(self) -> int:
        return sum(1 for t in self.map_tasks if t.result_accepted)

    @property
    def completed_reduce_tasks(self) -> int:
        return sum(1 for t in self.reduce_tasks if t.result_accepted)

    @property
    def map_progress(self) -> float:
        if not self.map_tasks:
            return 0.0
        unique_logical = len(set(t.logical_task_id for t in self.map_tasks))
        return self.completed_map_tasks / max(1, unique_logical)

    @property
    def reduce_progress(self) -> float:
        if not self.reduce_tasks:
            return 0.0
        unique_logical = len(set(t.logical_task_id for t in self.reduce_tasks))
        return self.completed_reduce_tasks / max(1, unique_logical)

    def all_maps_completed(self) -> bool:
        unique_logical = set(t.logical_task_id for t in self.map_tasks)
        return all(lid in self.accepted_logical_tasks for lid in unique_logical)

    def all_reduces_completed(self) -> bool:
        unique_logical = set(t.logical_task_id for t in self.reduce_tasks)
        return all(lid in self.accepted_logical_tasks for lid in unique_logical)

    def any_task_failed(self) -> bool:
        return any(t.state == TaskState.FAILED and t.attempt >= 4 for t in self.map_tasks + self.reduce_tasks)

    def get_accepted_map_tasks(self) -> List[Task]:
        return [t for t in self.map_tasks if t.result_accepted]

    def get_accepted_reduce_tasks(self) -> List[Task]:
        return [t for t in self.reduce_tasks if t.result_accepted]

    def generate_report(self) -> JobReport:
        self.report.map_task_reports = []
        self.report.reduce_task_reports = []

        seen_task_ids = set()
        for t in self.map_tasks + self.reduce_tasks:
            if t.task_id not in seen_task_ids:
                seen_task_ids.add(t.task_id)
                self.report.add_task_report(t)

        self.report.start_time = self.start_time
        self.report.end_time = self.end_time
        self.report.final_state = self.state.value
        self.report.total_duration = (self.end_time - self.start_time) if (self.end_time and self.start_time) else 0.0
        self.report.final_output_path = os.path.join(self.output_dir, f"result.{self.output_format}")
        return self.report


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
        output_dir: str = "./output",
        output_format: str = "text"
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
            output_dir=output_dir,
            output_format=output_format
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
            "num_map_tasks": len(set(t.logical_task_id for t in job.map_tasks)),
            "num_reduce_tasks": len(set(t.logical_task_id for t in job.reduce_tasks)),
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
        return [t for t in tasks if t.state == TaskState.PENDING and t.logical_task_id not in job.accepted_logical_tasks]

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

    def get_job_report(self, job_id: str) -> Optional[JobReport]:
        job = self.get_job(job_id)
        if not job:
            return None
        return job.generate_report()
