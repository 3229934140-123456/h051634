import os
import pickle
import hashlib
from typing import List, Dict, Any, Tuple
from collections import defaultdict


def hash_partition(key: Any, num_partitions: int) -> int:
    key_str = str(key)
    hash_val = int(hashlib.md5(key_str.encode()).hexdigest(), 16)
    return hash_val % num_partitions


def partition_map_output(
    map_output: List[Tuple[Any, Any]],
    num_partitions: int
) -> Dict[int, List[Tuple[Any, Any]]]:
    partitions: Dict[int, List[Tuple[Any, Any]]] = defaultdict(list)
    for key, value in map_output:
        part_id = hash_partition(key, num_partitions)
        partitions[part_id].append((key, value))
    return partitions


def sort_partition(partition: List[Tuple[Any, Any]]) -> List[Tuple[Any, Any]]:
    return sorted(partition, key=lambda x: str(x[0]))


def write_map_output_to_disk(
    partitions: Dict[int, List[Tuple[Any, Any]]],
    output_dir: str,
    task_id: str
) -> Dict[int, str]:
    output_files: Dict[int, str] = {}
    task_dir = os.path.join(output_dir, task_id)
    os.makedirs(task_dir, exist_ok=True)

    for part_id, data in partitions.items():
        sorted_data = sort_partition(data)
        file_path = os.path.join(task_dir, f"part-{part_id}.pickle")
        with open(file_path, "wb") as f:
            pickle.dump(sorted_data, f)
        output_files[part_id] = file_path

    return output_files


def read_map_output(file_path: str) -> List[Tuple[Any, Any]]:
    if not os.path.exists(file_path):
        return []
    with open(file_path, "rb") as f:
        return pickle.load(f)


def fetch_map_outputs(
    map_output_dirs: List[str],
    partition_id: int
) -> List[Tuple[Any, Any]]:
    all_data: List[Tuple[Any, Any]] = []

    for map_dir in map_output_dirs:
        part_file = os.path.join(map_dir, f"part-{partition_id}.pickle")
        if os.path.exists(part_file):
            data = read_map_output(part_file)
            all_data.extend(data)

    return all_data


def merge_sorted_lists(lists: List[List[Tuple[Any, Any]]]) -> List[Tuple[Any, Any]]:
    if not lists:
        return []

    result = []
    indices = [0] * len(lists)

    while True:
        min_val = None
        min_idx = -1

        for i, lst in enumerate(lists):
            if indices[i] < len(lst):
                current_val = lst[indices[i]]
                if min_val is None or str(current_val[0]) < str(min_val[0]):
                    min_val = current_val
                    min_idx = i

        if min_idx == -1:
            break

        result.append(min_val)
        indices[min_idx] += 1

    return result


def group_by_key(sorted_data: List[Tuple[Any, Any]]) -> List[Tuple[Any, List[Any]]]:
    grouped: List[Tuple[Any, List[Any]]] = []
    current_key = None
    current_values: List[Any] = []

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

    return grouped


class ShuffleManager:
    def __init__(self, base_output_dir: str):
        self.base_output_dir = base_output_dir

    def process_map_output(
        self,
        job_id: str,
        task_id: str,
        map_output: List[Tuple[Any, Any]],
        num_partitions: int
    ) -> Dict[int, str]:
        output_dir = os.path.join(self.base_output_dir, job_id, "map-outputs")
        partitions = partition_map_output(map_output, num_partitions)
        output_files = write_map_output_to_disk(partitions, output_dir, task_id)
        return output_files

    def get_partition_inputs(
        self,
        job_id: str,
        partition_id: int,
        map_task_ids: List[str]
    ) -> List[Tuple[Any, Any]]:
        map_output_dirs = []
        base_dir = os.path.join(self.base_output_dir, job_id, "map-outputs")

        for task_id in map_task_ids:
            task_dir = os.path.join(base_dir, task_id)
            if os.path.exists(task_dir):
                map_output_dirs.append(task_dir)

        return fetch_map_outputs(map_output_dirs, partition_id)

    def write_reduce_output(
        self,
        job_id: str,
        task_id: str,
        results: List[Tuple[Any, Any]]
    ) -> str:
        output_dir = os.path.join(self.base_output_dir, job_id, "reduce-outputs")
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{task_id}.pickle")
        with open(file_path, "wb") as f:
            pickle.dump(results, f)
        return file_path

    def read_reduce_output(self, job_id: str, task_id: str) -> List[Tuple[Any, Any]]:
        file_path = os.path.join(self.base_output_dir, job_id, "reduce-outputs", f"{task_id}.pickle")
        return read_map_output(file_path)

    def get_final_results(self, job_id: str, num_reduce_tasks: int) -> List[Tuple[Any, Any]]:
        all_results = []
        for i in range(num_reduce_tasks):
            task_id = f"{job_id}-r-{i}"
            try:
                results = self.read_reduce_output(job_id, task_id)
                all_results.extend(results)
            except Exception:
                pass
        return sorted(all_results, key=lambda x: str(x[0]))
