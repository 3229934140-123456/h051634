from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Dict, Optional, Tuple
import uuid
import time


class JobState(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    MAP_COMPLETED = "MAP_COMPLETED"
    REDUCE_COMPLETED = "REDUCE_COMPLETED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class TaskState(Enum):
    PENDING = "PENDING"
    ASSIGNED = "ASSIGNED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class TaskType(Enum):
    MAP = "MAP"
    REDUCE = "REDUCE"


@dataclass
class Task:
    task_id: str
    task_type: TaskType
    job_id: str
    logical_task_id: str
    state: TaskState = TaskState.PENDING
    worker_id: Optional[str] = None
    attempt: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    input_split: Optional[Any] = None
    partition_id: Optional[int] = None
    output_path: Optional[str] = None
    is_speculative: bool = False
    result_accepted: bool = False

    def mark_running(self, worker_id: str):
        self.state = TaskState.RUNNING
        self.worker_id = worker_id
        self.start_time = time.time()

    def mark_completed(self, output_path: str):
        self.state = TaskState.COMPLETED
        self.output_path = output_path
        self.end_time = time.time()

    def mark_failed(self):
        self.state = TaskState.FAILED
        self.end_time = time.time()

    @property
    def duration(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0


@dataclass
class InputSplit:
    split_id: str
    data: List[Any]
    start_idx: int
    length: int


@dataclass
class WorkerStatus:
    worker_id: str
    host: str
    port: int
    available: bool = True
    last_heartbeat: float = field(default_factory=time.time)
    num_map_slots: int = 2
    num_reduce_slots: int = 2
    running_tasks: List[str] = field(default_factory=list)

    @property
    def is_alive(self) -> bool:
        return time.time() - self.last_heartbeat < 30.0


def generate_id() -> str:
    return uuid.uuid4().hex[:12]
