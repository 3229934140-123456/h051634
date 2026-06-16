import os
import json
import pickle
import time
from typing import Dict, List, Optional, Any, Set
from .common import (
    JobState, TaskState, TaskType, Task,
    InputSplit, generate_id
)


class JobMetadataStore:
    """
    作业元数据持久化存储
    将作业状态、任务状态持久化到磁盘，支持重启恢复
    """

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def _get_job_dir(self, job_id: str) -> str:
        return os.path.join(self.base_dir, job_id)

    def _get_meta_path(self, job_id: str) -> str:
        return os.path.join(self._get_job_dir(job_id), "job_meta.json")

    def _get_tasks_path(self, job_id: str) -> str:
        return os.path.join(self._get_job_dir(job_id), "tasks.pickle")

    def save_job_meta(self, job):
        """保存作业元数据"""
        job_dir = self._get_job_dir(job.job_id)
        os.makedirs(job_dir, exist_ok=True)

        meta = {
            "job_id": job.job_id,
            "name": job.name,
            "state": job.state.value,
            "num_reduce_tasks": job.num_reduce_tasks,
            "num_map_tasks": job.num_map_tasks,
            "output_format": getattr(job, "output_format", "text"),
            "start_time": job.start_time,
            "end_time": job.end_time,
            "accepted_logical_tasks": list(job.accepted_logical_tasks),
            "map_outputs": {k: v for k, v in job.map_outputs.items()},
            "saved_at": time.time()
        }

        with open(self._get_meta_path(job.job_id), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def save_tasks(self, job):
        """保存所有任务状态"""
        job_dir = self._get_job_dir(job.job_id)
        os.makedirs(job_dir, exist_ok=True)

        tasks_data = {
            "map_tasks": [self._serialize_task(t) for t in job.map_tasks],
            "reduce_tasks": [self._serialize_task(t) for t in job.reduce_tasks]
        }

        with open(self._get_tasks_path(job.job_id), "wb") as f:
            pickle.dump(tasks_data, f)

    def save_all(self, job):
        """保存作业所有元数据"""
        self.save_job_meta(job)
        self.save_tasks(job)

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
            "result_accepted": task.result_accepted
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
        return task

    def load_job_meta(self, job_id: str) -> Optional[Dict]:
        """加载作业元数据"""
        meta_path = self._get_meta_path(job_id)
        if not os.path.exists(meta_path):
            return None

        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_tasks(self, job_id: str) -> Optional[Dict[str, List[Task]]]:
        """加载任务状态"""
        tasks_path = self._get_tasks_path(job_id)
        if not os.path.exists(tasks_path):
            return None

        with open(tasks_path, "rb") as f:
            data = pickle.load(f)

        return {
            "map_tasks": [self._deserialize_task(t) for t in data["map_tasks"]],
            "reduce_tasks": [self._deserialize_task(t) for t in data["reduce_tasks"]]
        }

    def list_jobs(self) -> List[Dict]:
        """列出所有历史作业"""
        jobs = []
        if not os.path.exists(self.base_dir):
            return jobs

        for entry in sorted(os.listdir(self.base_dir)):
            job_dir = os.path.join(self.base_dir, entry)
            meta_path = os.path.join(job_dir, "job_meta.json")
            if os.path.isdir(job_dir) and os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    if meta.get("job_id") == entry:
                        jobs.append({
                            "job_id": meta["job_id"],
                            "name": meta["name"],
                            "state": meta["state"],
                            "start_time": meta.get("start_time"),
                            "end_time": meta.get("end_time")
                        })
                except Exception:
                    pass

        return sorted(jobs, key=lambda x: x.get("start_time", 0), reverse=True)

    def job_exists(self, job_id: str) -> bool:
        """检查作业是否存在"""
        return os.path.exists(self._get_meta_path(job_id))

    def get_job_result_path(self, job_id: str, output_format: str = "text") -> Optional[str]:
        """获取作业结果文件路径"""
        result_path = os.path.join(self._get_job_dir(job_id), f"result.{output_format}")
        if os.path.exists(result_path):
            return result_path
        return None

    def get_job_report_path(self, job_id: str) -> Optional[str]:
        """获取作业报告路径"""
        report_path = os.path.join(self._get_job_dir(job_id), "job_report.json")
        if os.path.exists(report_path):
            return report_path
        return None

    def load_job_report(self, job_id: str) -> Optional[Dict]:
        """加载作业报告"""
        report_path = self.get_job_report_path(job_id)
        if not report_path:
            return None
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def cleanup_job(self, job_id: str) -> bool:
        """清理作业数据"""
        import shutil
        job_dir = self._get_job_dir(job_id)
        if os.path.exists(job_dir):
            shutil.rmtree(job_dir)
            return True
        return False
