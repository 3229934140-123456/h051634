#!/usr/bin/env python3
"""
精简版 MapReduce 框架命令行入口

使用示例:
  # 运行词频统计
  python cli.py run --input test_data/wordcount_input --job wordcount

  # 列出所有历史作业
  python cli.py history

  # 查看作业报告
  python cli.py report <job_id>

  # 恢复作业
  python cli.py resume --job-id <job_id> --input test_data/wordcount_input

  # 失败恢复演示
  python cli.py demo-recovery
"""

import sys
import os
import argparse
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mapreduce import MapReduceFramework


def word_count_map(line):
    words = line.strip().split()
    return [(word.lower(), 1) for word in words if word]


def word_count_reduce(key, values):
    return sum(values)


BUILTIN_JOBS = {
    "wordcount": {
        "map": word_count_map,
        "reduce": word_count_reduce,
        "description": "词频统计 - 统计文本中每个单词出现的次数"
    }
}


def _get_job_config(job_name):
    config = BUILTIN_JOBS.get(job_name)
    if not config:
        print(f"❌ 未知作业类型: {job_name}")
        print(f"可用作业: {', '.join(BUILTIN_JOBS.keys())}")
        return None
    return config


def cmd_run(args):
    """运行 MapReduce 作业"""
    print("=" * 70)
    print("🚀 启动精简版 MapReduce 集群")
    print("=" * 70)

    framework = MapReduceFramework(
        output_dir=args.output,
        num_workers=args.num_workers
    )

    print(f"\n📋 集群配置:")
    print(f"  工作节点数: {args.num_workers}")
    print(f"  输出目录: {args.output}")

    framework.start()
    print("\n✅ 集群启动成功")

    try:
        job_config = _get_job_config(args.job)
        if not job_config:
            return 1

        print(f"\n📦 提交作业: {args.job}")
        print(f"  描述: {job_config['description']}")
        print(f"  输入路径: {args.input}")
        print(f"  Map 任务数: {args.num_maps or '自动'}")
        print(f"  Reduce 任务数: {args.num_reduces}")
        print(f"  输出格式: {args.format}")
        print(f"  切分方式: {args.split_by}")

        if not os.path.exists(args.input):
            print(f"\n❌ 输入路径不存在: {args.input}")
            return 1

        job_id = framework.submit_job_from_files(
            name=args.job,
            input_dir=args.input,
            map_func=job_config["map"],
            reduce_func=job_config["reduce"],
            split_by=args.split_by,
            chunk_size=args.chunk_size,
            num_map_tasks=args.num_maps,
            num_reduce_tasks=args.num_reduces,
            output_format=args.format
        )

        print(f"\n🎯 作业已提交, Job ID: {job_id}")
        print("\n⏳ 执行进度:")

        status = framework.wait_for_job(job_id, timeout=args.timeout, show_progress=True)

        print(f"\n\n📊 作业执行结果:")
        print(f"  状态: {status['state']}")
        print(f"  Map 完成: {status['completed_maps']}/{status['num_map_tasks']}")
        print(f"  Reduce 完成: {status['completed_reduces']}/{status['num_reduce_tasks']}")

        if status["state"] == "SUCCEEDED":
            output_path = framework.finalize_job_output(job_id)
            results = framework.get_job_results(job_id)

            print(f"\n✅ 作业执行成功!")
            print(f"  结果条数: {len(results)}")
            print(f"  输出文件: {output_path}")

            if args.show_results:
                print(f"\n📋 结果预览 (前 {min(10, len(results))} 条):")
                print("-" * 50)
                for key, value in results[:10]:
                    print(f"  {key}\t{value}")
                if len(results) > 10:
                    print(f"  ... 还有 {len(results) - 10} 条")

            if args.show_report:
                framework.print_job_report(job_id)

            if args.show_files:
                job_dir = os.path.join(args.output, job_id)
                print(f"\n📁 输出目录结构: {job_dir}")
                _print_dir_tree(job_dir)

            return 0
        else:
            print(f"\n❌ 作业执行失败!")
            if args.show_report:
                framework.print_job_report(job_id)
            return 1

    except KeyboardInterrupt:
        print("\n\n⏹️  用户中断,正在停止集群...")
        return 130
    finally:
        print("\n🛑 正在停止集群...")
        framework.stop()
        print("✅ 集群已停止")


def cmd_resume(args):
    """恢复一个已存在的作业"""
    print("=" * 70)
    print("🔄 恢复作业运行")
    print("=" * 70)

    framework = MapReduceFramework(
        output_dir=args.output,
        num_workers=args.num_workers
    )

    if not framework.metadata_store.job_exists(args.job_id):
        print(f"\n❌ 作业不存在: {args.job_id}")
        return 1

    framework.start()

    try:
        job_config = _get_job_config(args.job)
        if not job_config:
            return 1

        print(f"\n📦 恢复作业: {args.job_id}")
        print(f"  输入路径: {args.input}")

        input_data = None
        if args.input:
            if not os.path.exists(args.input):
                print(f"❌ 输入路径不存在: {args.input}")
                return 1
            from mapreduce.shuffle import read_input_files
            input_data = read_input_files(args.input, args.split_by, args.chunk_size)

        restored_job_id = framework.resume_job(
            job_id=args.job_id,
            map_func=job_config["map"],
            reduce_func=job_config["reduce"],
            input_data=input_data
        )

        if not restored_job_id:
            print("❌ 恢复失败!")
            return 1

        job = framework.job_tracker.get_job(restored_job_id)
        if job:
            print(f"  当前状态: {job.state.value}")
            print(f"  恢复次数: {job.num_recoveries}")
            print(f"  已完成 Map: {job.completed_map_tasks}/{job.num_map_tasks}")
            print(f"  已完成 Reduce: {job.completed_reduce_tasks}/{job.num_reduce_tasks}")

            reused_maps = sum(1 for t in job.map_tasks if getattr(t, "is_reused", False))
            print(f"  复用的 Map 输出: {reused_maps} 个")

        print("\n⏳ 继续执行进度:")

        status = framework.wait_for_job(restored_job_id, timeout=args.timeout, show_progress=True)

        print(f"\n\n📊 最终结果:")
        print(f"  状态: {status['state']}")

        if status["state"] == "SUCCEEDED":
            output_path = framework.finalize_job_output(restored_job_id)
            results = framework.get_job_results(restored_job_id)

            print(f"\n✅ 作业恢复并完成!")
            print(f"  结果条数: {len(results)}")
            print(f"  输出文件: {output_path}")

            if args.show_report:
                framework.print_job_report(restored_job_id)

            return 0
        else:
            print(f"\n❌ 作业执行失败!")
            return 1

    except KeyboardInterrupt:
        print("\n\n⏹️  用户中断...")
        return 130
    finally:
        framework.stop()
        print("集群已停止")


def cmd_history(args):
    """列出所有历史作业（含空间占用）"""
    from mapreduce.metadata import JobMetadataStore
    store = JobMetadataStore(args.output)
    history_jobs = store.list_jobs_with_size()

    col_w = 15 if getattr(args, "verbose", False) else 90
    print("=" * col_w)
    header = f"📋 历史作业列表 (共 {len(history_jobs)} 个)"
    if not getattr(args, "verbose", False):
        print(header)
        print("=" * col_w)
        print(f"{'Job ID':<15} {'名称':<20} {'状态':<12} {'占用':<10} {'开始时间':<20} {'结束时间':<20}")
        print("-" * col_w)
    else:
        print(header)
        print("=" * col_w)
        print(f"{'Job ID':<15} {'名称':<20} {'状态':<12} {'占用':<10} {'开始时间':<20} {'结束时间':<20}")
        print("-" * col_w)

    for job in history_jobs:
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(job["start_time"])) if job.get("start_time") else "N/A"
        end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(job["end_time"])) if job.get("end_time") else "N/A"
        size = job.get("size", "0 B")
        print(f"{job['job_id']:<15} {job['name']:<20} {job['state']:<12} {size:<10} {start_time:<20} {end_time:<20}")

    if history_jobs:
        total_size = sum(j.get("size_bytes", 0) for j in history_jobs)
        print("-" * col_w)
        print(f"{'':15} {'':20} {'':12} 总占用: {store.format_size(total_size)}")

    print("=" * col_w)
    return 0


def cmd_result(args):
    """按 job id 查看结果文件内容"""
    from mapreduce.metadata import JobMetadataStore
    store = JobMetadataStore(args.output)

    if not store.job_exists(args.job_id):
        print(f"❌ 未找到作业: {args.job_id}")
        return 1

    max_lines = getattr(args, "lines", 0)
    fmt = getattr(args, "format", None)
    data = store.read_result_file(args.job_id, max_lines=max_lines, output_format=fmt)

    if not data:
        print(f"❌ 作业 {args.job_id} 暂无结果文件（作业可能未完成或结果已删除）")
        return 1

    print("=" * 80)
    print(f"📄 作业结果: {args.job_id}")
    print("=" * 80)
    print(f"  路径: {data['path']}")
    print(f"  格式: {data['format']}")
    print(f"  总行数: {data['total_lines']}")
    if data["truncated"]:
        print(f"  ⚠️  仅显示前 {max_lines} 行")
    print("-" * 80)
    for line in data["lines"]:
        print(f"  {line}")
    if data["truncated"]:
        print(f"  ... 还有 {data['total_lines'] - max_lines} 行，用 --lines 0 查看全部")
    print("=" * 80)
    return 0


def cmd_delete(args):
    """删除指定作业的元数据和输出文件"""
    from mapreduce.metadata import JobMetadataStore
    store = JobMetadataStore(args.output)

    if not store.job_exists(args.job_id):
        print(f"❌ 未找到作业: {args.job_id}")
        return 1

    size_before = store.get_job_size(args.job_id)
    size_str = store.format_size(size_before)

    if not args.yes:
        confirm = input(f"⚠️  即将删除作业 {args.job_id} (占用 {size_str})，此操作不可恢复！\n确认删除? [y/N] ").strip().lower()
        if confirm not in ("y", "yes"):
            print("已取消删除")
            return 0

    ok = store.delete_job(args.job_id)
    if ok:
        print(f"✅ 作业 {args.job_id} 已删除，释放空间 {size_str}")
        return 0
    else:
        print(f"❌ 删除作业 {args.job_id} 失败")
        return 1


def cmd_report(args):
    """查看作业报告"""
    framework = MapReduceFramework(output_dir=args.output, num_workers=1)
    framework.start()

    try:
        # 先尝试从内存加载（如果是活动作业）
        report = framework.get_job_report(args.job_id)

        if not report:
            # 从历史记录加载
            report_data = framework.metadata_store.load_job_report(args.job_id)
            if report_data:
                from mapreduce.job import JobReport
                report = JobReport(args.job_id, report_data.get("name", ""))
                report.load(report_data)

        if report:
            report.pretty_print()
            return 0
        else:
            print(f"❌ 未找到作业: {args.job_id}")
            return 1
    finally:
        framework.stop()


def cmd_status(args):
    """查看集群状态"""
    framework = MapReduceFramework(output_dir=args.output, num_workers=args.num_workers)
    framework.start()

    try:
        status = framework.get_cluster_status()
        print("=" * 60)
        print("📊 集群状态")
        print("=" * 60)
        print(f"  工作节点数: {status['num_workers']}")
        print(f"  活跃节点数: {status['active_workers']}")
        print(f"  总 Map 槽位: {status['total_map_slots']}")
        print(f"  总 Reduce 槽位: {status['total_reduce_slots']}")
        print(f"  运行中任务数: {status['running_tasks']}")
        print(f"\n  作业统计:")
        print(f"    总数: {status['jobs']['total']}")
        print(f"    运行中: {status['jobs']['running']}")
        print(f"    已完成: {status['jobs']['completed']}")
        print(f"    失败: {status['jobs']['failed']}")
        print("=" * 60)
    finally:
        framework.stop()


def cmd_demo_recovery(args):
    """失败恢复演示"""
    print("=" * 70)
    print("🧪 失败恢复演示")
    print("=" * 70)
    print("\n选择演示类型:")
    print("  1. Worker 失败恢复 (同进程内模拟)")
    print("  2. 进程重启恢复 (模拟进程退出后恢复)")

    if args.type == "worker":
        choice = "1"
    elif args.type == "restart":
        choice = "2"
    else:
        choice = input("\n请输入选项 (默认 1): ").strip() or "1"

    from examples.demo_recovery import demo_worker_failure_recovery, demo_process_restart_recovery

    if choice == "1":
        success = demo_worker_failure_recovery()
    elif choice == "2":
        success = demo_process_restart_recovery()
    else:
        print("无效选项")
        return 1

    return 0 if success else 1


def _print_dir_tree(path, prefix=""):
    """打印目录树"""
    if not os.path.isdir(path):
        return

    entries = sorted(os.listdir(path))
    for i, entry in enumerate(entries):
        full_path = os.path.join(path, entry)
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "

        if os.path.isdir(full_path):
            print(f"{prefix}{connector}{entry}/")
            extension = "    " if is_last else "│   "
            _print_dir_tree(full_path, prefix + extension)
        else:
            size = os.path.getsize(full_path)
            size_str = f" ({size} bytes)" if size < 1024 else f" ({size//1024} KB)"
            print(f"{prefix}{connector}{entry}{size_str}")


def main():
    parser = argparse.ArgumentParser(
        description="精简版 MapReduce 分布式批处理框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行词频统计作业
  python cli.py run --input test_data/wordcount_input --job wordcount

  # 运行作业并显示结果和报告
  python cli.py run --input test_data/wordcount_input --job wordcount --show-results --show-report

  # 列出所有历史作业
  python cli.py history

  # 查看作业报告
  python cli.py report <job_id>

  # 恢复作业
  python cli.py resume --job-id <job_id> --input test_data/wordcount_input

  # 失败恢复演示
  python cli.py demo-recovery
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # run 命令
    run_parser = subparsers.add_parser("run", help="运行 MapReduce 作业")
    run_parser.add_argument("--input", "-i", required=True, help="输入文件或目录路径")
    run_parser.add_argument("--output", "-o", default="./output", help="输出目录 (默认: ./output)")
    run_parser.add_argument("--job", "-j", default="wordcount", help="作业类型 (默认: wordcount)")
    run_parser.add_argument("--num-workers", "-w", type=int, default=3, help="工作节点数 (默认: 3)")
    run_parser.add_argument("--num-maps", "-m", type=int, default=0, help="Map 任务数 (0=自动)")
    run_parser.add_argument("--num-reduces", "-r", type=int, default=2, help="Reduce 任务数 (默认: 2)")
    run_parser.add_argument("--format", "-f", choices=["text", "jsonl"], default="text", help="输出格式 (默认: text)")
    run_parser.add_argument("--split-by", choices=["lines", "size", "files"], default="lines", help="输入切分方式 (默认: lines)")
    run_parser.add_argument("--chunk-size", type=int, default=100, help="分片大小 (行数或字节数, 默认: 100)")
    run_parser.add_argument("--timeout", type=int, default=300, help="超时时间(秒) (默认: 300)")
    run_parser.add_argument("--show-results", action="store_true", help="显示结果预览")
    run_parser.add_argument("--show-report", action="store_true", help="显示详细作业报告")
    run_parser.add_argument("--show-files", action="store_true", help="显示输出目录结构")

    # resume 命令
    resume_parser = subparsers.add_parser("resume", help="恢复一个已存在的作业")
    resume_parser.add_argument("--job-id", required=True, help="作业 ID")
    resume_parser.add_argument("--input", "-i", default=None, help="输入文件或目录路径 (可选)")
    resume_parser.add_argument("--output", "-o", default="./output", help="输出目录 (默认: ./output)")
    resume_parser.add_argument("--job", "-j", default="wordcount", help="作业类型 (默认: wordcount)")
    resume_parser.add_argument("--num-workers", "-w", type=int, default=3, help="工作节点数")
    resume_parser.add_argument("--split-by", choices=["lines", "size", "files"], default="lines", help="输入切分方式")
    resume_parser.add_argument("--chunk-size", type=int, default=100, help="分片大小")
    resume_parser.add_argument("--timeout", type=int, default=300, help="超时时间(秒)")
    resume_parser.add_argument("--show-report", action="store_true", help="显示详细作业报告")

    # history 命令
    hist_parser = subparsers.add_parser("history", help="列出所有历史作业")
    hist_parser.add_argument("--output", "-o", default="./output", help="输出目录")

    # report 命令
    report_parser = subparsers.add_parser("report", help="查看作业报告")
    report_parser.add_argument("job_id", help="作业 ID")
    report_parser.add_argument("--output", "-o", default="./output", help="输出目录")

    # status 命令
    status_parser = subparsers.add_parser("status", help="查看集群状态")
    status_parser.add_argument("--output", "-o", default="./output", help="输出目录")
    status_parser.add_argument("--num-workers", "-w", type=int, default=3, help="工作节点数")

    # result 命令
    result_parser = subparsers.add_parser("result", help="查看作业结果内容")
    result_parser.add_argument("job_id", help="作业 ID")
    result_parser.add_argument("--output", "-o", default="./output", help="输出目录")
    result_parser.add_argument("--lines", "-n", type=int, default=0,
                               help="显示前 N 行，0 表示全部 (默认: 0)")
    result_parser.add_argument("--format", "-f", choices=["text", "jsonl"], default=None,
                               help="结果格式（默认自动匹配）")

    # delete 命令
    delete_parser = subparsers.add_parser("delete", help="删除作业的元数据和输出文件")
    delete_parser.add_argument("job_id", help="作业 ID")
    delete_parser.add_argument("--output", "-o", default="./output", help="输出目录")
    delete_parser.add_argument("--yes", "-y", action="store_true", help="跳过确认提示")

    # demo-recovery 命令
    demo_parser = subparsers.add_parser("demo-recovery", help="失败恢复演示")
    demo_parser.add_argument("--type", "-t", choices=["worker", "restart"],
                             default=None, help="演示类型: worker 或 restart")
    demo_parser.add_argument("--output", "-o", default="./output", help="输出目录")

    args = parser.parse_args()

    if args.command == "run":
        return cmd_run(args)
    elif args.command == "resume":
        return cmd_resume(args)
    elif args.command == "history":
        return cmd_history(args)
    elif args.command == "report":
        return cmd_report(args)
    elif args.command == "result":
        return cmd_result(args)
    elif args.command == "delete":
        return cmd_delete(args)
    elif args.command == "status":
        return cmd_status(args)
    elif args.command == "demo-recovery":
        return cmd_demo_recovery(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
