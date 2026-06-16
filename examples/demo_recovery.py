import sys
import os
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mapreduce import MapReduceFramework
from mapreduce.common import TaskState


def slow_word_count_map(line):
    """带延迟的 map 函数，便于观察进度"""
    time.sleep(0.05)
    words = line.strip().split()
    return [(word.lower(), 1) for word in words if word]


def word_count_reduce(key, values):
    return sum(values)


def demo_process_restart_recovery():
    print("\n" + "="*70)
    print("🔬 演示: 进程重启后恢复作业 (Process Restart Recovery)")
    print("="*70)
    print("\n说明:")
    print("  1. 第一次运行: 启动作业，运行 1 秒后强制停止 (模拟进程崩溃)")
    print("  2. 第二次运行: 从磁盘加载作业状态，跳过已完成的 map，继续剩余任务")
    print("  3. 验证: 结果正确，报告显示哪些分片是复用的、哪些是重跑的")

    input_data = []
    for i in range(100):
        input_data.append(f"demo line number {i} hello world mapreduce test data")
        input_data.append(f"another line {i} with recovery test words")

    expected_counts = {}
    for line in input_data:
        for word in line.strip().split():
            w = word.lower()
            expected_counts[w] = expected_counts.get(w, 0) + 1

    print(f"\n输入数据: {len(input_data)} 行")
    print(f"预期唯一单词数: {len(expected_counts)}")

    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")

    # 清理之前的测试
    import shutil
    test_output = os.path.join(output_dir, "restart-demo")
    if os.path.exists(test_output):
        shutil.rmtree(test_output)
    os.makedirs(test_output, exist_ok=True)

    job_id = None

    # ===== 第一次运行 =====
    print("\n" + "-"*70)
    print("📌 第一次运行: 启动作业，1.5 秒后强制停止 (模拟进程崩溃)")
    print("-"*70)

    framework1 = MapReduceFramework(output_dir=test_output, num_workers=2)
    framework1.start()

    try:
        job_id = framework1.submit_job(
            name="restart-recovery-demo",
            input_data=input_data,
            map_func=slow_word_count_map,
            reduce_func=word_count_reduce,
            num_map_tasks=12,
            num_reduce_tasks=2,
            output_format="text"
        )

        print(f"作业已提交: {job_id}")
        print("运行中... (1.2 秒后模拟崩溃)")

        time.sleep(1.2)

        job = framework1.job_tracker.get_job(job_id)
        if job:
            job.save_metadata()
            
            # 保存崩溃快照：确保恢复时从这一刻的状态开始
            import shutil
            snapshot_dir = os.path.join(test_output, f"{job_id}_snapshot")
            if os.path.exists(snapshot_dir):
                shutil.rmtree(snapshot_dir)
            shutil.copytree(job.output_dir, snapshot_dir)
            
            completed_maps = job.completed_map_tasks
            total_maps = job.num_map_tasks
            state = job.state.value
            print(f"\n崩溃时状态: {state}")
            print(f"已完成 Map: {completed_maps}/{total_maps}")
            print(f"已完成 Reduce: {job.completed_reduce_tasks}/{job.num_reduce_tasks}")
            print("📸 已保存崩溃时刻的状态快照")

    finally:
        framework1.stop()
        print("\n💥 第一次进程结束 (模拟崩溃)")

    # ===== 第二次运行: 恢复 =====
    print("\n" + "-"*70)
    print("� 第二次运行: 从磁盘恢复作业，继续运行")
    print("-"*70)

    framework2 = MapReduceFramework(output_dir=test_output, num_workers=2)
    framework2.start()

    try:
        # 恢复崩溃快照：确保从崩溃时的状态开始恢复
        import shutil
        snapshot_dir = os.path.join(test_output, f"{job_id}_snapshot")
        job_dir = os.path.join(test_output, job_id)
        if os.path.exists(snapshot_dir) and os.path.exists(job_dir):
            shutil.rmtree(job_dir)
            shutil.copytree(snapshot_dir, job_dir)

        restored_job_id = framework2.resume_job(
            job_id=job_id,
            map_func=slow_word_count_map,
            reduce_func=word_count_reduce,
            input_data=input_data
        )

        if not restored_job_id:
            print("❌ 恢复失败!")
            return False

        job = framework2.job_tracker.get_job(restored_job_id)
        if job:
            print(f"作业已恢复: {restored_job_id}")
            print(f"  恢复次数: {job.num_recoveries}")
            print(f"  当前阶段: {job.state.value}")
            print(f"  已完成 Map: {job.completed_map_tasks}/{job.num_map_tasks}")
            print(f"  已完成 Reduce: {job.completed_reduce_tasks}/{job.num_reduce_tasks}")

            reused_maps = sum(1 for t in job.map_tasks if getattr(t, "is_reused", False) and t.result_accepted)
            print(f"  ♻️  复用 Map 输出: {reused_maps} 个 (跳过重算)")
            print(f"  🔄 待重跑 Map: {job.num_map_tasks - reused_maps} 个")

        print("\n继续运行直到完成...")
        status = framework2.wait_for_job(restored_job_id, timeout=60, show_progress=True)

        print(f"\n最终状态: {status['state']}")

        if status["state"] == "SUCCEEDED":
            output_path = framework2.finalize_job_output(restored_job_id)
            results = framework2.get_job_results(restored_job_id)
            result_dict = {k: v for k, v in results}

            all_correct = all(result_dict.get(k, 0) == v for k, v in expected_counts.items())

            print(f"\n{'='*70}")
            if all_correct:
                print("✅ 验证通过: 恢复后结果正确!")
            else:
                print("❌ 验证失败: 结果不正确!")
                for k, v in expected_counts.items():
                    actual = result_dict.get(k, 0)
                    if actual != v:
                        print(f"  {k}: 预期 {v}, 实际 {actual}")
            print(f"最终输出文件: {output_path}")

            report = framework2.get_job_report(restored_job_id)
            if report:
                stats = report.recovery_stats
                print(f"\n📊 恢复统计:")
                print(f"  恢复次数: {stats.get('num_recoveries', 0)}")
                print(f"  复用 Map 输出: {stats.get('reused_map_outputs', 0)} 个")
                print(f"  复用 Reduce 输出: {stats.get('reused_reduce_outputs', 0)} 个")
                print(f"  重跑 Map 任务: {stats.get('rerun_map_tasks', 0)} 个")
                print(f"  重跑 Reduce 任务: {stats.get('rerun_reduce_tasks', 0)} 个")

                print(f"\n📦 Map 任务详情:")
                print(f"  {'任务ID':<28} {'状态':<6} {'类型':<8} {'Worker':<10}")
                print(f"  {'-'*56}")
                for r in report.map_task_reports:
                    state = "✅ 采纳" if r["result_accepted"] else "   "
                    if r.get("is_reused", False):
                        task_type = "♻️ 复用"
                    elif r["is_speculative"]:
                        task_type = "⚡ 推测"
                    else:
                        task_type = "🔄 新跑"
                    print(f"  {r['task_id']:<28} {state:<6} {task_type:<8} {str(r['worker_id']):<10}")
                    if r.get("partition_files") and r["result_accepted"]:
                        n = len(r["partition_files"])
                        print(f"    ↳ 写出 {n} 个分区文件")

            return all_correct

        return False

    finally:
        framework2.stop()
        print("第二次运行结束")


def demo_worker_failure_recovery():
    print("\n" + "="*70)
    print("🔬 演示: Worker 失败恢复 (Worker Failure Recovery)")
    print("="*70)
    print("\n说明:")
    print("  1. 启动 3 个 worker 的集群，运行一个作业")
    print("  2. 运行中途，模拟 worker-1 死亡")
    print("  3. 观察容错机制检测到死亡，并重新调度任务")
    print("  4. 最终验证结果正确")

    input_data = []
    for i in range(80):
        input_data.append(f"worker failure test line {i} hello mapreduce")
        input_data.append(f"another test line {i} data recovery fault")

    expected_counts = {}
    for line in input_data:
        for word in line.strip().split():
            w = word.lower()
            expected_counts[w] = expected_counts.get(w, 0) + 1

    print(f"\n输入数据: {len(input_data)} 行")
    print(f"预期唯一单词数: {len(expected_counts)}")

    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")

    import shutil
    test_output = os.path.join(output_dir, "worker-fail-demo")
    if os.path.exists(test_output):
        shutil.rmtree(test_output)

    framework = MapReduceFramework(output_dir=test_output, num_workers=3)
    framework.start()

    try:
        job_id = framework.submit_job(
            name="worker-failure-demo",
            input_data=input_data,
            map_func=slow_word_count_map,
            reduce_func=word_count_reduce,
            num_map_tasks=8,
            num_reduce_tasks=2,
            output_format="text"
        )

        print(f"作业已提交: {job_id}")
        print("运行 1 秒后模拟 worker-1 失败...")

        time.sleep(1.0)

        print("\n� 模拟 worker-1 失败!")
        framework.simulate_worker_failure("worker-1")

        print("等待容错处理和任务重新调度...")

        status = framework.wait_for_job(job_id, timeout=60, show_progress=True)

        print(f"\n最终状态: {status['state']}")

        if status["state"] == "SUCCEEDED":
            output_path = framework.finalize_job_output(job_id)
            results = framework.get_job_results(job_id)
            result_dict = {k: v for k, v in results}

            all_correct = all(result_dict.get(k, 0) == v for k, v in expected_counts.items())

            print(f"\n{'='*70}")
            if all_correct:
                print("✅ 验证通过: Worker 失败恢复后结果正确!")
            else:
                print("❌ 验证失败: 结果不正确!")
            print(f"最终输出文件: {output_path}")

            report = framework.get_job_report(job_id)
            if report:
                print(f"\n📊 执行统计:")
                print(f"  总 Map 尝试: {report.summary['total_map_tasks']} 次")
                print(f"  采纳结果: {report.summary['accepted_map_results']} 个")
                print(f"  失败尝试: {report.summary['failed_attempts']} 次")
                print(f"  推测执行: {report.summary['speculative_attempts']} 次")

                print(f"\n📦 Map 任务详情 (前 5 个):")
                for r in report.map_task_reports[:5]:
                    state = "✅" if r["result_accepted"] else "❌"
                    worker = r["worker_id"] or "N/A"
                    print(f"  {r['task_id']:<25} {state} worker={worker:<10} 耗时={r['duration']:.2f}s")

            return all_correct

        return False

    finally:
        framework.stop()
        print("\n框架已停止")


def main():
    print("="*70)
    print("🧪 容错与恢复功能演示")
    print("="*70)

    print("\n选择演示:")
    print("  1. 进程重启恢复 (模拟进程退出后恢复)")
    print("  2. Worker 失败恢复 (同进程内模拟)")
    print("  3. 全部运行")

    choice = input("\n请输入选项 (默认 1): ").strip() or "1"

    success1 = True
    success2 = True

    if choice in ("1", "3"):
        success1 = demo_process_restart_recovery()

    if choice in ("2", "3"):
        success2 = demo_worker_failure_recovery()

    print("\n" + "="*70)
    print("演示总结:")
    if choice in ("1", "3"):
        print(f"  进程重启恢复: {'✅ 成功' if success1 else '❌ 失败'}")
    if choice in ("2", "3"):
        print(f"  Worker 失败恢复: {'✅ 成功' if success2 else '❌ 失败'}")
    print("="*70)


if __name__ == "__main__":
    main()
