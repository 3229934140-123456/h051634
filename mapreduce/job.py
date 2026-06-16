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
        self.recovery_stats: Dict[str, Any] = {
            "num_recoveries": 0,
            "reused_map_outputs": 0,
            "reused_reduce_outputs": 0,
            "rerun_map_tasks": 0,
            "rerun_reduce_tasks": 0
        }

    def add_task_report(self, task: Task, partition_files: Optional[Dict[int, str]] = None,
                        fetched_map_files: Optional[List[str]] = None):
        report = {
            "task_id": task.task_id,
            "logical_task_id": task.logical_task_id,
            "task_type": task.task_type.value,
            "state": task.state.value,
            "worker_id": task.worker_id,
            "attempt": task.attempt,
            "is_speculative": task.is_speculative,
            "result_accepted": task.result_accepted,
            "is_reused": getattr(task, "is_reused", False),
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
            "partition_id": task.partition_id,
            "partition_files": partition_files if partition_files else {},
            "fetched_map_files": fetched_map_files if fetched_map_files else []
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
            "speculative_attempts": sum(1 for r in self.map_task_reports + self.reduce_task_reports if r["is_speculative"]),
            "reused_map_outputs": sum(1 for r in self.map_task_reports if r.get("is_reused", False) and r["result_accepted"]),
            "reused_reduce_outputs": sum(1 for r in self.reduce_task_reports if r.get("is_reused", False) and r["result_accepted"])
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
            "recovery_stats": self.recovery_stats,
            "map_tasks": self.map_task_reports,
            "reduce_tasks": self.reduce_task_reports,
            "summary": self.summary
        }

    def save(self, output_dir: str):
        report_path = os.path.join(output_dir, "job_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def load(self, report_dict: Dict):
        self.map_task_reports = report_dict.get("map_tasks", [])
        self.reduce_task_reports = report_dict.get("reduce_tasks", [])
        self.start_time = report_dict.get("start_time")
        self.end_time = report_dict.get("end_time")
        self.final_state = report_dict.get("final_state")
        self.final_output_path = report_dict.get("final_output_path")
        self.total_duration = report_dict.get("total_duration", 0.0)
        self.recovery_stats = report_dict.get("recovery_stats", self.recovery_stats)

    def pretty_print(self):
        print(f"\n{'='*70}")
        print(f"📊 作业运行报告: {self.name}")
        print(f"{'='*70}")
        print(f"作业ID: {self.job_id}")
        print(f"最终状态: {self.final_state}")
        print(f"总耗时: {self.total_duration:.2f} 秒")
        print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time)) if self.start_time else 'N/A'}")
        print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.end_time)) if self.end_time else 'N/A'}")
        print(f"最终输出: {self.final_output_path}")

        if self.recovery_stats.get("num_recoveries", 0) > 0:
            print(f"\n🔄 恢复统计:")
            print(f"  恢复次数: {self.recovery_stats['num_recoveries']}")
            print(f"  复用 Map 输出: {self.recovery_stats['reused_map_outputs']} 个")
            print(f"  复用 Reduce 输出: {self.recovery_stats['reused_reduce_outputs']} 个")
            print(f"  重跑 Map 任务: {self.recovery_stats['rerun_map_tasks']} 个")
            print(f"  重跑 Reduce 任务: {self.recovery_stats['rerun_reduce_tasks']} 个")

        print(f"\n📋 概览:")
        print(f"  Map 任务尝试: {self.summary['total_map_tasks']} 次 (成功采纳: {self.summary['accepted_map_results']})")
        print(f"  Reduce 任务尝试: {self.summary['total_reduce_tasks']} 次 (成功采纳: {self.summary['accepted_reduce_results']})")
        print(f"  失败尝试: {self.summary['failed_attempts']} 次")
        print(f"  推测执行: {self.summary['speculative_attempts']} 次")

        print(f"\n📦 Map 任务详情:")
        print(f"{'任务ID':<32} {'采纳':<5} {'推测':<5} {'复用':<5} {'Worker':<10} {'耗时(s)':<10} {'重试':<6}")
        print(f"{'-'*78}")
        for r in self.map_task_reports:
            spec = "是" if r["is_speculative"] else "否"
            accepted = "✅" if r["result_accepted"] else "❌"
            reused = "♻️" if r.get("is_reused", False) else " "
            print(f"{r['task_id']:<32} {accepted:<5} {spec:<5} {reused:<5} {str(r['worker_id']):<10} {r['duration']:<10.2f} {r['attempt']:<6}")
            if r["input_split"]:
                s = r["input_split"]
                print(f"  ↳ 分片: {s['start_idx']}-{s['start_idx']+s['length']} ({s['num_records']} 条)")
            if r.get("partition_files"):
                print(f"  ↳ 分区输出 ({len(r['partition_files'])} 个):")
                for pid, fpath in sorted(r["partition_files"].items()):
                    print(f"      part-{pid}: {fpath}")

        print(f"\n🔧 Reduce 任务详情:")
        print(f"{'任务ID':<32} {'采纳':<5} {'推测':<5} {'复用':<5} {'Worker':<10} {'耗时(s)':<10} {'重试':<6}")
        print(f"{'-'*78}")
        for r in self.reduce_task_reports:
            spec = "是" if r["is_speculative"] else "否"
            accepted = "✅" if r["result_accepted"] else "❌"
            reused = "♻️" if r.get("is_reused", False) else " "
            print(f"{r['task_id']:<32} {accepted:<5} {spec:<5} {reused:<5} {str(r['worker_id']):<10} {r['duration']:<10.2f} {r['attempt']:<6}")
            print(f"  ↳ 分区: {r['partition_id']}, 输出: {r['output_path']}")
            if r.get("fetched_map_files"):
                print(f"  ↳ 拉取 Map 文件 ({len(r['fetched_map_files'])} 个):")
                for fpath in r["fetched_map_files"][:5]:
                    print(f"      {fpath}")
                if len(r["fetched_map_files"]) > 5:
                    print(f"      ... 还有 {len(r['fetched_map_files'])-5} 个")
        print(f"{'='*70}\n")


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
        self.map_partition_files: Dict[str, Dict[int, str]] = {}
        self.accepted_logical_tasks: Set[str] = set()
        self.report = JobReport(job_id, name)
        self.num_recoveries: int = 0

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
                if t.task_type == TaskType.MAP:
                    part_files = self.map_partition_files.get(t.task_id, {})
                    self.report.add_task_report(t, partition_files=part_files)
                else:
                    fetched = getattr(t, "fetched_map_files", None)
                    self.report.add_task_report(t, fetched_map_files=fetched)

        self.report.start_time = self.start_time
        self.report.end_time = self.end_time
        self.report.final_state = self.state.value
        self.report.total_duration = (self.end_time - self.start_time) if (self.end_time and self.start_time) else 0.0
        self.report.final_output_path = os.path.join(self.output_dir, f"result.{self.output_format}")
        self.report.recovery_stats["num_recoveries"] = self.num_recoveries

        return self.report

    def save_metadata(self):
        """持久化作业元数据到磁盘"""
        meta_path = os.path.join(self.output_dir, "job_meta.json")
        meta = {
            "job_id": self.job_id,
            "name": self.name,
            "state": self.state.value,
            "num_map_tasks": self.num_map_tasks,
            "num_reduce_tasks": self.num_reduce_tasks,
            "output_format": self.output_format,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "accepted_logical_tasks": list(self.accepted_logical_tasks),
            "map_outputs": self.map_outputs,
            "map_partition_files": {k: {str(pk): pv for pk, pv in v.items()}
                                    for k, v in self.map_partition_files.items()},
            "num_recoveries": self.num_recoveries,
            "saved_at": time.time()
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        tasks_path = os.path.join(self.output_dir, "tasks.pickle")
        tasks_data = {
            "map_tasks": [],
            "reduce_tasks": []
        }
        for t in self.map_tasks:
            tasks_data["map_tasks"].append(self._serialize_task(t))
        for t in self.reduce_tasks:
            tasks_data["reduce_tasks"].append(self._serialize_task(t))
        with open(tasks_path, "wb") as f:
            pickle.dump(tasks_data, f)

    def _serialize_task(self, task: Task) -> Dict:
        return {
            "task_id": task.task_id,
            "task_type": task.task_type.value,
            "job_id": task.job_id,
            "logical_task_id": task.logical_task_id,
            "state": task.state.value,
            "worker_id": task.worker_id,
            "attempt": task.attempt,
            "start_time": task.start_time,
            "end_time": task.end_time,
            "input_split": {
                "split_id": task.input_split.split_id,
                "start_idx": task.input_split.start_idx,
                "length": task.input_split.length,
                "data": task.input_split.data
            } if task.input_split else None,
            "partition_id": task.partition_id,
            "output_path": task.output_path,
            "is_speculative": task.is_speculative,
            "result_accepted": task.result_accepted,
            "is_reused": getattr(task, "is_reused", False),
            "fetched_map_files": getattr(task, "fetched_map_files", [])
        }

    def _deserialize_task(self, data: Dict) -> Task:
        task = Task(
            task_id=data["task_id"],
            task_type=TaskType(data["task_type"]),
            job_id=data["job_id"],
            logical_task_id=data["logical_task_id"],
            state=TaskState(data["state"]),
            worker_id=data.get("worker_id"),
            attempt=data.get("attempt", 0),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            input_split=InputSplit(
                split_id=data["input_split"]["split_id"],
                data=data["input_split"]["data"],
                start_idx=data["input_split"]["start_idx"],
                length=data["input_split"]["length"]
            ) if data.get("input_split") else None,
            partition_id=data.get("partition_id"),
            output_path=data.get("output_path"),
            is_speculative=data.get("is_speculative", False),
            result_accepted=data.get("result_accepted", False)
        )
        if data.get("is_reused", False):
            setattr(task, "is_reused", True)
        fetched = data.get("fetched_map_files", [])
        if fetched:
            setattr(task, "fetched_map_files", fetched)
        return task

    @classmethod
    def load_from_disk(cls, job_id: str, output_dir: str,
                       map_func: Callable, reduce_func: Callable,
                       input_data: Optional[List[Any]] = None) -> Optional['Job']:
        """从磁盘加载作业，用于恢复运行"""
        job_dir = os.path.join(output_dir, job_id)
        meta_path = os.path.join(job_dir, "job_meta.json")
        tasks_path = os.path.join(job_dir, "tasks.pickle")

        if not os.path.exists(meta_path):
            return None

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        job = cls.__new__(cls)
        job.job_id = meta["job_id"]
        job.name = meta["name"]
        job.input_data = input_data or []
        job.map_func = map_func
        job.reduce_func = reduce_func
        job.num_map_tasks = meta["num_map_tasks"]
        job.num_reduce_tasks = meta["num_reduce_tasks"]
        job.output_dir = job_dir
        job.output_format = meta.get("output_format", "text")
        job.state = JobState(meta["state"])
        job.start_time = meta.get("start_time")
        job.end_time = meta.get("end_time")
        job.map_outputs = meta.get("map_outputs", {})
        job.map_partition_files = {}
        mpfs = meta.get("map_partition_files", {})
        for k, v in mpfs.items():
            job.map_partition_files[k] = {int(pk): pv for pk, pv in v.items()}
        job.accepted_logical_tasks = set(meta.get("accepted_logical_tasks", []))
        job.num_recoveries = meta.get("num_recoveries", 0) + 1
        job.report = JobReport(job.job_id, job.name)

        report_path = os.path.join(job_dir, "job_report.json")
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                report_dict = json.load(f)
            job.report.load(report_dict)

        if os.path.exists(tasks_path):
            with open(tasks_path, "rb") as f:
                tasks_data = pickle.load(f)
            job.map_tasks = [job._deserialize_task(t) for t in tasks_data["map_tasks"]]
            job.reduce_tasks = [job._deserialize_task(t) for t in tasks_data["reduce_tasks"]]
        else:
            job.map_tasks = []
            job.reduce_tasks = []

        return job

    def prepare_for_resume(self) -> Dict[str, int]:
        """准备恢复运行，重置未完成的任务状态，返回恢复统计"""
        stats = {
            "reused_map_outputs": 0,
            "reused_reduce_outputs": 0,
            "rerun_map_tasks": 0,
            "rerun_reduce_tasks": 0
        }

        if self.state == JobState.SUCCEEDED:
            return stats

        if self.state in (JobState.MAP_COMPLETED, JobState.REDUCE_COMPLETED,
                          JobState.RUNNING, JobState.PENDING):
            pass

        for task in self.map_tasks:
            if task.result_accepted:
                stats["reused_map_outputs"] += 1
                setattr(task, "is_reused", True)
            elif task.state in (TaskState.ASSIGNED, TaskState.RUNNING):
                task.state = TaskState.PENDING
                task.worker_id = None
                task.start_time = None
                task.end_time = None
                stats["rerun_map_tasks"] += 1
            elif task.state == TaskState.PENDING:
                stats["rerun_map_tasks"] += 1
            elif task.state == TaskState.FAILED and not task.is_speculative:
                if task.attempt < 4:
                    task.state = TaskState.PENDING
                    task.worker_id = None
                    stats["rerun_map_tasks"] += 1

        for task in self.reduce_tasks:
            if task.result_accepted:
                stats["reused_reduce_outputs"] += 1
                setattr(task, "is_reused", True)
            elif task.state in (TaskState.ASSIGNED, TaskState.RUNNING):
                task.state = TaskState.PENDING
                task.worker_id = None
                task.start_time = None
                task.end_time = None
                stats["rerun_reduce_tasks"] += 1
            elif task.state == TaskState.PENDING:
                stats["rerun_reduce_tasks"] += 1
            elif task.state == TaskState.FAILED and not task.is_speculative:
                if task.attempt < 4:
                    task.state = TaskState.PENDING
                    task.worker_id = None
                    stats["rerun_reduce_tasks"] += 1

        if self.state == JobState.MAP_COMPLETED:
            stats["rerun_map_tasks"] = 0
        elif self.state in (JobState.REDUCE_COMPLETED, JobState.SUCCEEDED):
            stats["rerun_map_tasks"] = 0
            stats["rerun_reduce_tasks"] = 0

        self.report.recovery_stats["num_recoveries"] = self.num_recoveries
        self.report.recovery_stats["reused_map_outputs"] = stats["reused_map_outputs"]
        self.report.recovery_stats["reused_reduce_outputs"] = stats["reused_reduce_outputs"]
        self.report.recovery_stats["rerun_map_tasks"] = stats["rerun_map_tasks"]
        self.report.recovery_stats["rerun_reduce_tasks"] = stats["rerun_reduce_tasks"]

        self.state = JobState.RUNNING

        return stats


class JobTracker:
    def __init__(self, base_output_dir: str = "./output"):
        self.base_output_dir = base_output_dir
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
        job.save_metadata()
        return job_id

    def resume_job(
        self,
        job_id: str,
        map_func: Callable,
        reduce_func: Callable,
        input_data: Optional[List[Any]] = None,
        output_dir: str = "./output"
    ) -> Optional[str]:
        """恢复一个已存在的作业"""
        job = Job.load_from_disk(job_id, output_dir, map_func, reduce_func, input_data)
        if not job:
            return None

        stats = job.prepare_for_resume()
        self.jobs[job_id] = job
        if job_id not in self.job_queue:
            self.job_queue.append(job_id)
        job.save_metadata()
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
            "end_time": job.end_time,
            "num_recoveries": job.num_recoveries
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
            job.save_metadata()

    def save_job(self, job_id: str):
        """立即保存作业状态"""
        job = self.get_job(job_id)
        if job:
            job.save_metadata()

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
