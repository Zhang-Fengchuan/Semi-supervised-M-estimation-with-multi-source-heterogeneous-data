"""
数值模拟唯一主入口
==================

用途
----
本脚本用于复现论文正文和附录中的数值模拟结果。平时只需要打开本文件，
修改开头的 TARGET、USE_EXISTING_RESULTS、RUN_SIMULATION 和 T，然后运行。

重要约定
--------
1. Example 1--4 均使用统一 MST/MDSP/交集式选择流程。
2. Example 4 不使用 z-band 选择器。
3. 本整理版不迁移旧结果；重新模拟后结果会保存到“模拟结果”下对应实验文件夹。
"""

from __future__ import annotations

import shutil
import sys
import time
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# =============================================================================
# 1. 平时最常修改的参数
# =============================================================================

# 可选：
#   "Table2"   : 正文实验一，6 个异质无标签源，N=5000
#   "Table3"   : 正文实验三，同质无标签源，N=5000
#   "Table4"   : 正文实验四，高阶异质源，N=5000
#   "Figure3"  : 正文实验二，36 个无标签源，选择频次和信息集图所需数据
#   "FigureS4" : MST 剪枝路径相关数据，默认使用实验二设置
#   "ALL_MAIN" : 跑正文主要表格和图形数据
#   "ALL"      : 跑当前整理版支持的全部目标
TARGET = "Table2"

# True  表示读取“模拟结果”里已有的最新结果，只重新整理表格或查看文件。
# False 表示不读取已有结果。
USE_EXISTING_RESULTS = False

# True  表示重新运行模拟。
# False 表示不重新运行；如果 USE_EXISTING_RESULTS=True，则只读取已有结果。
RUN_SIMULATION = True

# 正式论文结果一般设为 500；调试时可改为 1 或 2。
T = 500

# 工作模型：正文最终模拟使用 "linear"；逻辑回归敏感性分析可改为 "logistic"。
MODEL = "linear"

# 有标签样本量、无标签样本量和维度。
N0_VALUES = [250, 500, 1000, 2000]
NK = 5000
P = 5

# 随机种子。
SEED = 123


# =============================================================================
# 2. 最终算法口径
# =============================================================================

# 线性回归正文口径。quad1 表示真实生成模型包含二次项，但工作模型只用线性项。
LINEAR_DGP = "quad1"
LINEAR_X_DISTRIBUTION = "single_gaussian"
LINEAR_INTERCEPT_FROM_SUPERVISED = True
LINEAR_BIAS_CORRECTION = False

# 逻辑回归统一口径。默认只供敏感性分析使用。
LOGISTIC_SS_SOLVER = "bfgs"
LOGISTIC_SUP_SOLVER = "bfgs"
LOGISTIC_OPTIM_TOLERANCE = 5e-3
LOGISTIC_OPTIM_MAX_ITER = 2000
LOGISTIC_BETA_STAR_TOLERANCE = 1e-6
LOGISTIC_BETA_STAR_MAX_ITER = 1000


# =============================================================================
# 3. 路径和目标映射
# =============================================================================

THIS_FILE = Path(__file__).resolve()
ROOT_DIR = THIS_FILE.parents[1]
CORE_DIR = ROOT_DIR / "核心函数"
RESULT_ROOT = THIS_FILE.parent / "模拟结果"

for path in (THIS_FILE.parent, CORE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import _simulation_engine as engine  # noqa: E402


TARGET_MAP: Dict[str, Dict[str, object]] = {
    "Table2": {
        "example": "exm1",
        "folder": "实验一_六个异质无标签源",
        "file_prefix": "表2_实验一_N5000_T{T}",
        "description": "正文 Table 2：Example 1，K=6，N=5000。",
    },
    "Table3": {
        "example": "exm3",
        "folder": "实验三_同质无标签源",
        "file_prefix": "表3_实验三_同质_N5000_T{T}",
        "description": "正文 Table 3：Example 3，同质无标签源，N=5000。",
    },
    "Table4": {
        "example": "exm4",
        "folder": "实验四_高阶异质源",
        "file_prefix": "表4_实验四_高阶异质_N5000_T{T}",
        "description": "正文 Table 4：Example 4，高阶异质源，N=5000。",
    },
    "Figure3": {
        "example": "exm2",
        "folder": "实验二_三十六个无标签源",
        "file_prefix": "图3_实验二_选择频次数据_N5000_T{T}",
        "description": "正文 Figure 3：Example 2，36 个无标签源的信息集和选择频次。",
    },
    "FigureS4": {
        "example": "exm2",
        "folder": "实验二_三十六个无标签源",
        "file_prefix": "图S4_MST剪枝路径数据_N5000_T{T}",
        "description": "附录 Figure S4：MST 剪枝路径相关数据，默认沿用 Example 2。",
    },
}

TARGET_GROUPS = {
    "ALL_MAIN": ["Table2", "Figure3", "Table3", "Table4"],
    "ALL_SUPPLEMENT": ["FigureS4"],
    "ALL": ["Table2", "Figure3", "Table3", "Table4", "FigureS4"],
}


def configure_engine() -> None:
    """把本主入口的参数写入内部调度器。"""
    engine.config.SEED = SEED
    engine.config.LINEAR_DGP = LINEAR_DGP
    engine.config.LINEAR_X_DISTRIBUTION = LINEAR_X_DISTRIBUTION
    engine.config.LINEAR_INTERCEPT_FROM_SUPERVISED = LINEAR_INTERCEPT_FROM_SUPERVISED
    engine.config.LINEAR_BIAS_CORRECTION = LINEAR_BIAS_CORRECTION
    engine.config.LOGISTIC_SS_SOLVER = LOGISTIC_SS_SOLVER
    engine.config.LOGISTIC_SUP_SOLVER = LOGISTIC_SUP_SOLVER
    engine.config.LOGISTIC_OPTIM_TOLERANCE = LOGISTIC_OPTIM_TOLERANCE
    engine.config.LOGISTIC_OPTIM_MAX_ITER = LOGISTIC_OPTIM_MAX_ITER
    engine.config.LOGISTIC_BETA_STAR_TOLERANCE = LOGISTIC_BETA_STAR_TOLERANCE
    engine.config.LOGISTIC_BETA_STAR_MAX_ITER = LOGISTIC_BETA_STAR_MAX_ITER


def expand_targets(target: str) -> List[str]:
    """把 ALL_MAIN / ALL_SUPPLEMENT / ALL 展开成具体图表编号。"""
    target = str(target)
    if target in TARGET_GROUPS:
        return TARGET_GROUPS[target]
    if target not in TARGET_MAP:
        raise ValueError(f"未知 TARGET={target!r}，可选 {sorted(TARGET_MAP) + sorted(TARGET_GROUPS)}")
    return [target]


def latest_result_dir(experiment_dir: Path, model: str, example: str) -> Path | None:
    """找到某个实验文件夹下最新一次模拟输出目录。"""
    if not experiment_dir.exists():
        return None
    candidates = [
        path for path in experiment_dir.iterdir()
        if path.is_dir() and f"_{model}_{example}_T" in path.name
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def copy_named_outputs(run_dir: Path, experiment_dir: Path, file_prefix: str) -> List[Path]:
    """
    把内部输出目录中的 summary/selection 文件复制成中文清晰文件名。

    原始逐参数结果和 meta 文件仍保留在带时间戳的 run_dir 中，方便复查。
    """
    generated: List[Path] = []
    prefix = file_prefix.format(T=T)
    copy_plan = {
        "summary_all.csv": f"{prefix}.csv",
        "summary_all.tex": f"{prefix}.tex",
        "selection_counts_all.csv": f"{prefix}_选择频次.csv",
        "selection_counts_all.md": f"{prefix}_选择频次.md",
        "run_log.json": f"{prefix}_运行日志.json",
    }
    for src_name, dst_name in copy_plan.items():
        src = run_dir / src_name
        if src.exists():
            dst = experiment_dir / dst_name
            shutil.copy2(src, dst)
            generated.append(dst)
    return generated


def make_figure3_from_selection_counts(run_dir: Path, experiment_dir: Path, file_prefix: str) -> List[Path]:
    """
    根据 Example 2 的 selection_counts_all.csv 生成 Figure 3 的基础 PNG。

    该图使用整理版 SelectionViz 中的多面板函数。若本机没有绘图依赖，模拟结果
    仍然可用，只会跳过 PNG 生成。
    """
    selection_path = run_dir / "selection_counts_all.csv"
    if not selection_path.exists():
        return []

    try:
        from SelectionViz import plot_multi_panel
    except Exception as exc:
        print(f"无法导入 SelectionViz，跳过 Figure 3 图片生成：{exc!r}")
        return []

    df = pd.read_csv(selection_path)
    h_mu = [-0.01, 0.01, -0.5, 0.5, -1, 1]
    h_sigma = [0.01, 0.01, 0.5, 0.5, 1, 1]
    results = {}
    for n0 in sorted(df["n0"].unique()):
        block = df[df["n0"] == n0]
        matrix = np.zeros((6, 6), dtype=float)
        for _, row in block.iterrows():
            match = re.match(r"m(\d+)s(\d+)", str(row["source"]))
            if match:
                i = int(match.group(1)) - 1
                j = int(match.group(2)) - 1
                if 0 <= i < 6 and 0 <= j < 6:
                    matrix[i, j] = float(row["select_count"])
        results[int(n0)] = (matrix, h_mu, h_sigma)

    if not results:
        return []

    try:
        out_png = Path(plot_multi_panel(
            results,
            which_Exm=2,
            sample_size_N=NK,
            simulation_times=T,
            code_dir=str(experiment_dir),
        ))
    except Exception as exc:
        print(f"Figure 3 图片生成失败，但数值结果已保存：{exc!r}")
        return []

    dst = experiment_dir / f"{file_prefix.format(T=T)}.png"
    shutil.copy2(out_png, dst)
    return [dst]


def show_existing_outputs(
    experiment_dir: Path,
    model: str,
    example: str,
    file_prefix: str,
    make_figure3: bool = False,
) -> List[Path]:
    """读取已有结果时，显示并重新整理最新输出。"""
    run_dir = latest_result_dir(experiment_dir, model=model, example=example)
    if run_dir is None:
        print(f"没有找到已有结果：{experiment_dir}")
        return []
    print(f"读取已有结果目录：{run_dir}")
    generated = copy_named_outputs(run_dir, experiment_dir, file_prefix)
    summary_path = run_dir / "summary_all.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        print(summary[["method", "n0", "SE", "SSE", "MSE", "MRR", "CP", "TIME"]].to_string(index=False))
    if make_figure3:
        generated.extend(make_figure3_from_selection_counts(run_dir, experiment_dir, file_prefix))
    return generated


def run_target(target: str) -> List[Path]:
    """运行或读取单个图表目标。"""
    info = TARGET_MAP[target]
    example = str(info["example"])
    experiment_dir = RESULT_ROOT / str(info["folder"])
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"当前目标：{target}")
    print(f"说明：{info['description']}")
    print(f"模型：{MODEL}, T={T}, n0={N0_VALUES}, N={NK}, p={P}")
    print(f"读取已有结果：{USE_EXISTING_RESULTS}")
    print(f"重新模拟：{RUN_SIMULATION}")
    print(f"实验输出文件夹：{experiment_dir}")

    generated: List[Path] = []
    if USE_EXISTING_RESULTS:
        generated.extend(
            show_existing_outputs(
                experiment_dir,
                model=MODEL,
                example=example,
                file_prefix=str(info["file_prefix"]),
                make_figure3=(target == "Figure3"),
            )
        )
        if generated and not RUN_SIMULATION:
            return generated

    if not RUN_SIMULATION:
        print("RUN_SIMULATION=False，且没有可用已有结果，本目标未重新运行。")
        return generated

    run_dir = engine.run_simulation(
        model=MODEL,
        example=example,
        T=T,
        n0_values=N0_VALUES,
        N=NK,
        p=P,
        out_root=experiment_dir,
        quiet=False,
    )
    generated.extend(copy_named_outputs(run_dir, experiment_dir, str(info["file_prefix"])))
    if target == "Figure3":
        generated.extend(make_figure3_from_selection_counts(run_dir, experiment_dir, str(info["file_prefix"])))
    print("整理后的输出文件：")
    for path in generated:
        print(f"  - {path}")
    return generated


def main() -> None:
    """主程序。"""
    started = time.time()
    configure_engine()
    targets = expand_targets(TARGET)
    all_outputs: List[Path] = []
    for target in targets:
        all_outputs.extend(run_target(target))
    elapsed = time.time() - started

    print("=" * 80)
    print("全部目标处理完成。")
    print(f"总用时：{elapsed:.1f} 秒")
    print("生成或整理出的文件：")
    for path in all_outputs:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
