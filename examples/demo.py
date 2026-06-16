import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mapreduce import MapReduceFramework


def word_count_map(line):
    words = line.strip().split()
    return [(word.lower(), 1) for word in words if word]


def word_count_reduce(key, values):
    return sum(values)


def inverted_index_map(doc):
    doc_id, content = doc
    words = content.strip().split()
    return [(word.lower(), doc_id) for word in set(words) if word]


def inverted_index_reduce(key, values):
    return sorted(list(set(values)))


def max_temperature_map(record):
    year, temp = record
    return [(year, temp)]


def max_temperature_reduce(key, values):
    return max(values)


def demo_word_count(framework):
    print("\n" + "="*60)
    print("示例 1: 词频统计 (Word Count)")
    print("="*60)

    input_data = [
        "Hello world hello mapreduce",
        "MapReduce is a distributed programming model",
        "Hello distributed computing world",
        "Programming model for big data",
        "Big data processing with MapReduce",
        "Hello big data world",
        "Distributed computing is powerful",
        "MapReduce model is simple yet powerful"
    ]

    print(f"\n输入数据: {len(input_data)} 行文本")

    job_id = framework.submit_job(
        name="word-count-demo",
        input_data=input_data,
        map_func=word_count_map,
        reduce_func=word_count_reduce,
        num_map_tasks=3,
        num_reduce_tasks=2
    )

    print(f"作业已提交: {job_id}")
    print("等待作业完成...")

    status = framework.wait_for_job(job_id, timeout=60)
    print(f"\n作业状态: {status['state']}")
    print(f"Map 进度: {status['map_progress']*100:.1f}%")
    print(f"Reduce 进度: {status['reduce_progress']*100:.1f}%")

    if status["state"] == "SUCCEEDED":
        results = framework.get_job_results(job_id)
        print(f"\n结果 ({len(results)} 个单词):")
        print("-" * 40)
        for word, count in results:
            print(f"  {word:20s} -> {count}")

    return status["state"] == "SUCCEEDED"


def demo_inverted_index(framework):
    print("\n" + "="*60)
    print("示例 2: 倒排索引 (Inverted Index)")
    print("="*60)

    documents = [
        ("doc1", "Hello world hello mapreduce"),
        ("doc2", "MapReduce is a distributed programming model"),
        ("doc3", "Hello distributed computing world"),
        ("doc4", "Programming model for big data"),
        ("doc5", "Big data processing with MapReduce")
    ]

    print(f"\n输入数据: {len(documents)} 个文档")

    job_id = framework.submit_job(
        name="inverted-index-demo",
        input_data=documents,
        map_func=inverted_index_map,
        reduce_func=inverted_index_reduce,
        num_map_tasks=2,
        num_reduce_tasks=2
    )

    print(f"作业已提交: {job_id}")
    print("等待作业完成...")

    status = framework.wait_for_job(job_id, timeout=60)
    print(f"\n作业状态: {status['state']}")

    if status["state"] == "SUCCEEDED":
        results = framework.get_job_results(job_id)
        print(f"\n结果 ({len(results)} 个词条):")
        print("-" * 50)
        for word, doc_ids in results:
            print(f"  {word:20s} -> {doc_ids}")

    return status["state"] == "SUCCEEDED"


def demo_max_temperature(framework):
    print("\n" + "="*60)
    print("示例 3: 最高气温 (Max Temperature)")
    print("="*60)

    temperature_data = [
        (2020, 35), (2020, 32), (2020, 38), (2020, 28),
        (2021, 36), (2021, 39), (2021, 33), (2021, 31),
        (2022, 40), (2022, 37), (2022, 35), (2022, 42),
        (2023, 38), (2023, 36), (2023, 39), (2023, 34),
        (2024, 41), (2024, 43), (2024, 38), (2024, 40)
    ]

    print(f"\n输入数据: {len(temperature_data)} 条气温记录")

    job_id = framework.submit_job(
        name="max-temperature-demo",
        input_data=temperature_data,
        map_func=max_temperature_map,
        reduce_func=max_temperature_reduce,
        num_map_tasks=2,
        num_reduce_tasks=2
    )

    print(f"作业已提交: {job_id}")
    print("等待作业完成...")

    status = framework.wait_for_job(job_id, timeout=60)
    print(f"\n作业状态: {status['state']}")

    if status["state"] == "SUCCEEDED":
        results = framework.get_job_results(job_id)
        print(f"\n结果 (每年最高气温):")
        print("-" * 30)
        for year, temp in results:
            print(f"  {year} 年 -> {temp}°C")

    return status["state"] == "SUCCEEDED"


def demo_cluster_status(framework):
    print("\n" + "="*60)
    print("集群状态")
    print("="*60)

    status = framework.get_cluster_status()
    print(f"  工作节点数: {status['num_workers']}")
    print(f"  活跃节点数: {status['active_workers']}")
    print(f"  总 Map 槽位: {status['total_map_slots']}")
    print(f"  总 Reduce 槽位: {status['total_reduce_slots']}")
    print(f"  运行中任务数: {status['running_tasks']}")
    print(f"  作业总数: {status['jobs']['total']}")
    print(f"    运行中: {status['jobs']['running']}")
    print(f"    已完成: {status['jobs']['completed']}")
    print(f"    失败: {status['jobs']['failed']}")


def main():
    print("="*60)
    print("精简版 MapReduce 分布式计算框架演示")
    print("="*60)

    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    framework = MapReduceFramework(output_dir=output_dir, num_workers=3)

    print(f"\n启动框架 (3 个工作节点)...")
    framework.start()
    time.sleep(0.5)

    demo_cluster_status(framework)

    try:
        success1 = demo_word_count(framework)
        print(f"\n词频统计示例: {'成功' if success1 else '失败'}")

        success2 = demo_inverted_index(framework)
        print(f"\n倒排索引示例: {'成功' if success2 else '失败'}")

        success3 = demo_max_temperature(framework)
        print(f"\n最高气温示例: {'成功' if success3 else '失败'}")

        print("\n" + "="*60)
        print("所有示例执行完毕")
        print(f"成功: {sum([success1, success2, success3])}/3")
        print("="*60)

        demo_cluster_status(framework)

    finally:
        print("\n停止框架...")
        framework.stop()
        print("框架已停止")


if __name__ == "__main__":
    main()
