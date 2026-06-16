#!/usr/bin/env python3
"""
精简版 MapReduce 框架命令行入口

使用示例:
  # 启动本地集群并运行词频统计
  python cli.py run --input test_data/wordcount_input --output output --job wordcount

  # 查看集群状态
  python cli.py status

  # 查看作业报告
  python cli.py report <job_id>

  # 列出所有作业
  python cli.py list
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


def cmd_run(args):
    """运行 MapReduce 作业"""
    print("=" * 60)
    print("🚀 启动精简版 MapReduce 集群")
    print("=" * 60)

    framework = MapReduceFramework(
        output_dir=args.output,
        num_workers=args.num_workers
    )

    print(f"\n📋 集群配置:")
    print(f"  工作节点数: {args.num_workers}")
    print(f"  每个节点 Map 槽位: 2")
    print(f"  每个节点 Reduce 槽位: 2")
    print(f"  输出目录: {args.output}")

    framework.start()
    print("\n✅ 集群启动成功")

    try:
        job_config = BUILTIN_JOBS.get(args.job)
        if not job_config:
            print(f"\n❌ 未知作业类型: {args.job}")
            print(f"可用作业: {', '.join(BUILTIN_JOBS.keys())}")
            return 1

        print(f"\n📦 提交作业: {args.job}")
        print(f"  描述: {job_config['description']}")
        print(f"  输入路径: {args.input}")
        print(f"  Map 任务数: {args.num_maps or '自动'}")
        print(f"  Reduce 任务数: {args.num_reduces}")
        print(f"  输出格式: {args.format}")
        print(f"  切分方式: {args.split_by}")
        if args.split_by in ("lines", "size"):
            print(f"  分片大小: {args.chunk_size}")

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
                print("-" * 40)
                for key, value in results[:10]:
                    print(f"  {key}\t{value}")
                if len(results) > 10:
                    print(f"  ... 还有 {len(results) - 10} 条")

            if args.show_report:
                framework.print_job_report(job_id)

            print(f"\n📁 输出目录结构: {os.path.join(args.output, job_id)}")
            for root, dirs, files in os.walk(os.path.join(args.output, job_id)):
                level = root.replace(os.path.join(args.output, job_id), '').count(os.sep)
                indent = ' ' * 2 * level
                print(f'{indent}{os.path.basename(root)}/')
                subindent = ' ' * 2 * (level + 1)
                for file in files:
                    filepath = os.path.join(root, file)
                    size = os.path.getsize(filepath)
                    print(f'{subindent}{file} ({size} bytes)')

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


def cmd_list(args):
    """列出所有作业"""
    framework = MapReduceFramework(output_dir=args.output, num_workers=1)
    framework.start()

    try:
        jobs = framework.list_jobs()
        print("=" * 80)
        print(f"{'Job ID':<15} {'名称':<20} {'状态':<15} {'Map进度':<10} {'Reduce进度':<10}")
        print("-" * 80)
        for job in jobs:
            map_p = f"{job['map_progress']*100:.0f}%"
            reduce_p = f"{job['reduce_progress']*100:.0f}%"
            print(f"{job['job_id']:<15} {job['name']:<20} {job['state']:<15} {map_p:<10} {reduce_p:<10}")
        print("=" * 80)
    finally:
        framework.stop()


def cmd_report(args):
    """查看作业运行报告"""
    framework = MapReduceFramework(output_dir=args.output, num_workers=1)
    framework.start()

    try:
        report = framework.get_job_report(args.job_id)
        if report:
            report.pretty_print()
        else:
            print(f"❌ 未找到作业: {args.job_id}")
            return 1
    finally:
        framework.stop()

    return 0


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

  # 使用 JSONL 输出格式
  python cli.py run --input test_data/wordcount_input --job wordcount --format jsonl

  # 查看集群状态
  python cli.py status

  # 列出所有作业
  python cli.py list

  # 查看作业报告
  python cli.py report <job_id>
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

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

    status_parser = subparsers.add_parser("status", help="查看集群状态")
    status_parser.add_argument("--output", "-o", default="./output", help="输出目录")
    status_parser.add_argument("--num-workers", "-w", type=int, default=3, help="工作节点数")

    list_parser = subparsers.add_parser("list", help="列出所有作业")
    list_parser.add_argument("--output", "-o", default="./output", help="输出目录")

    report_parser = subparsers.add_parser("report", help="查看作业运行报告")
    report_parser.add_argument("job_id", help="作业 ID")
    report_parser.add_argument("--output", "-o", default="./output", help="输出目录")

    args = parser.parse_args()

    if args.command == "run":
        return cmd_run(args)
    elif args.command == "status":
        return cmd_status(args)
    elif args.command == "list":
        return cmd_list(args)
    elif args.command == "report":
        return cmd_report(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
