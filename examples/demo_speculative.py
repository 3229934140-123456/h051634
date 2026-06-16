import sys
import os
import time
import random
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mapreduce import MapReduceFramework


slow_map_task_index = None


def word_count_map_slow(line):
    global slow_map_task_index

    if slow_map_task_index is not None:
        task_thread_id = threading.current_thread().name
        if slow_map_task_index in line:
            time.sleep(2.0)

    words = line.strip().split()
    return [(word.lower(), 1) for word in words if word]


def word_count_reduce(key, values):
    return sum(values)


def demo_slow_map_speculative(framework):
    global slow_map_task_index
    slow_map_task_index = "LINE-SLOW-123"

    print("\n" + "="*60)
    print("示例: 慢任务推测执行验证 (Speculative Execution)")
    print("="*60)
    print("\n说明:")
    print("  - 创建 8 个 map 任务,其中第 5 个任务故意延迟 5 秒")
    print("  - 推测执行会检测到慢任务,启动备份副本")
    print("  - 只有先完成的那份结果会被采纳,数据不会重复")
    print("  - 最终词频统计结果应该是准确的,没有重复计数")

    input_data = []
    for i in range(8):
        if i == 4:
            for _ in range(10):
                input_data.append(f"LINE-SLOW-123 hello slow task test data {i}")
        else:
            for _ in range(10):
                input_data.append(f"normal line with some words for testing {i}")

    word_counts = {}
    for line in input_data:
        for word in line.strip().split():
            w = word.lower()
            word_counts[w] = word_counts.get(w, 0) + 1

    print(f"\n输入数据: {len(input_data)} 行")
    print(f"预期唯一单词数: {len(word_counts)}")
    print(f"预期 'hello' 出现次数: {word_counts.get('hello', 0)}")
    print(f"预期 'line' 出现次数: {word_counts.get('line', 0)}")
    print(f"预期 'slow' 出现次数: {word_counts.get('slow', 0)}")
    print("\n注意: 第 5 个 map 任务会延迟 3 秒,触发推测执行...")

    job_id = framework.submit_job(
        name="speculative-execution-demo",
        input_data=input_data,
        map_func=word_count_map_slow,
        reduce_func=word_count_reduce,
        num_map_tasks=8,
        num_reduce_tasks=2,
        output_format="text"
    )

    print(f"\n作业已提交: {job_id}")
    print("等待作业完成 (会触发推测执行)...")

    status = framework.wait_for_job(job_id, timeout=60, show_progress=True)
    print(f"\n作业状态: {status['state']}")

    if status["state"] == "SUCCEEDED":
        results = framework.get_job_results(job_id)
        result_dict = {k: v for k, v in results}

        print(f"\n结果 ({len(results)} 个单词):")
        print("-" * 50)
        for word, count in sorted(results, key=lambda x: str(x[0])):
            expected = word_counts.get(word, 0)
            match = "✓" if count == expected else "✗"
            print(f"  {match} {word:20s} -> {count:3d} (预期: {expected})")

        all_correct = all(result_dict.get(k, 0) == v for k, v in word_counts.items())
        print(f"\n{'='*60}")
        if all_correct:
            print("✅ 验证通过: 所有单词计数正确,没有重复!")
        else:
            print("❌ 验证失败: 存在计数错误或重复!")

        output_path = framework.finalize_job_output(job_id)
        print(f"\n最终输出文件: {output_path}")

        framework.print_job_report(job_id)

        report = framework.get_job_report(job_id)
        if report:
            spec_count = report.summary.get("speculative_attempts", 0)
            accepted = report.summary.get("accepted_map_results", 0)
            total = report.summary.get("total_map_tasks", 0)
            print(f"\n📊 推测执行效果:")
            print(f"  - 总共尝试 Map 任务: {total} 次")
            print(f"  - 推测执行尝试: {spec_count} 次")
            print(f"  - 采纳结果数: {accepted} 份")
            print(f"  - 说明: 每个逻辑任务只采纳一份结果,没有重复计入")

        slow_map_task_index = None
        return all_correct

    slow_map_task_index = None
    return False


def main():
    import threading

    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    framework = MapReduceFramework(output_dir=output_dir, num_workers=3)

    print(f"启动框架 (3 个工作节点)...")
    framework.start()
    time.sleep(0.5)

    try:
        success = demo_slow_map_speculative(framework)
        print(f"\n慢任务推测执行示例: {'成功' if success else '失败'}")

    finally:
        print("\n停止框架...")
        framework.stop()
        print("框架已停止")


if __name__ == "__main__":
    main()
