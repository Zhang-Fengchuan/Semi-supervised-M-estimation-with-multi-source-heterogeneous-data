import numpy as np
import pandas as pd
from scipy.stats import norm
import random
import sys
import os
from pathlib import Path
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

# 允许直接运行本文件。
# 本文件位于“实际数据分析/”，核心半监督算法位于上层“核心函数/”。
# 如果不手动加入 core 路径，PyCharm 直接运行本脚本时会报：
# ModuleNotFoundError: No module named 'MstMdsp'
THIS_FILE = Path(__file__).resolve()
ROOT_DIR = THIS_FILE.parents[1]
CORE_DIR = ROOT_DIR / "核心函数"
for path in (ROOT_DIR, CORE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from ModelSpec import LogisticModelSpec

# ======================================== 默认实际数据配置 ========================================
# PyCharm 直接运行本文件时，默认使用当前正文实际数据分析口径：
#   1. 使用原始 PC1-FPCA 特征，并采用随机划分 seed=509 生成的训练/测试/无标签数据；
#   2. 使用原始逻辑回归损失，不启用类别平衡权重；
#   3. 不启用统一 L2 正则化；
#   4. 半监督方法的截距项不额外替换为监督估计；
#   5. 分类阈值固定为 0.5。
# 如果以后需要做敏感性分析，可以在 PyCharm/终端环境变量中手动设置
# REALDATA_BEIJING_BASE、REALDATA_DEYANG_BASE、REALDATA_CLASS_WEIGHT、
# REALDATA_L2_LAMBDA、REALDATA_INTERCEPT_FROM_SUPERVISED 等参数。

# 是否只读取已有结果。若设为 True 且 RUN_ANALYSIS=False，本脚本只打印已有结果表。
USE_EXISTING_RESULTS = os.environ.get(
    "REALDATA_USE_EXISTING_RESULTS", "0"
).strip().lower() in {"1", "true", "yes", "y"}

# 是否重新从原始数据读取、筛选、估计并输出完整结果。
RUN_ANALYSIS = os.environ.get(
    "REALDATA_RUN_ANALYSIS", "1"
).strip().lower() in {"1", "true", "yes", "y"}

# 结果输出目录。默认保存在本整理版“实际数据分析/分析结果”中。
RESULT_DIR = Path(os.environ.get(
    "REALDATA_RESULT_DIR",
    str(THIS_FILE.parent / "分析结果" / "实际数据_原始PC1_seed509")
))

if USE_EXISTING_RESULTS and not RUN_ANALYSIS:
    metrics_path = RESULT_DIR / "prediction_metrics.csv"
    coef_path = RESULT_DIR / "coefficient_pvalues.csv"
    print(f"读取已有实际数据结果：{RESULT_DIR}")
    if metrics_path.exists():
        print("\n预测指标：")
        print(pd.read_csv(metrics_path).to_string(index=False))
    else:
        print(f"未找到预测指标表：{metrics_path}")
    if coef_path.exists():
        print("\n变量显著性表：")
        print(pd.read_csv(coef_path, header=None).to_string(index=False))
    else:
        print(f"未找到变量显著性表：{coef_path}")
    sys.exit(0)

# ======================================== 基础设置 ========================================
# 设置随机种子（对应MATLAB的rng(1,'philox')），保证结果可复现
np.random.seed(1)
random.seed(1)

# ======================================== 加载北京医院数据 ========================================
# 真实步态视频和由视频生成的特征文件不随公开代码发布。
# 运行实际数据分析前，请在本机环境变量中指定两个数据目录，例如：
#   export REALDATA_BEIJING_BASE="/path/to/beijing/outcome_output_dir/..."
#   export REALDATA_DEYANG_BASE="/path/to/deyang/outcome_output_dir/..."
def _required_data_path(env_name: str) -> str:
    """读取实际数据目录；未设置时给出明确提示，避免公开代码暴露本地敏感路径。"""
    value = os.environ.get(env_name, "").strip()
    if not value:
        raise RuntimeError(
            f"请先设置环境变量 {env_name}。真实步态视频数据及其特征文件不随公开代码发布。"
        )
    return value


base_path_beijing = _required_data_path("REALDATA_BEIJING_BASE")
base_path_beijing = base_path_beijing.rstrip("/") + "/"

# 读取带标签的训练/测试数据和无标签数据。
# 保留 X 的列名，用于后面输出完整的系数/p 值表；这对 PC2/PC3 等多分量 FPCA 尤其重要。
X_labeled_train_df = pd.read_csv(f"{base_path_beijing}单任务_X_labeled_train.csv", encoding="gbk")
feature_names = X_labeled_train_df.columns.astype(str).tolist()
X_labeled_train = X_labeled_train_df.values
Y_labeled_train = pd.read_csv(f"{base_path_beijing}单任务_Y_labeled_train.csv", encoding="gbk").values
X_labeled_test = pd.read_csv(f"{base_path_beijing}单任务_X_labeled_test.csv", encoding="gbk").values
Y_labeled_test = pd.read_csv(f"{base_path_beijing}单任务_Y_labeled_test.csv", encoding="gbk").values
X_unlabeled = pd.read_csv(f"{base_path_beijing}单任务_X_unlabeled.csv", encoding="gbk").values

# 重塑数据维度（增加通道维度，对应MATLAB的reshape(...,1)）
# X_unlabeled = np.reshape(X_unlabeled, (X_unlabeled.shape[0], X_unlabeled.shape[1], 1))

# 获取数据集维度信息
sample_size_n, p = X_labeled_train.shape  # 训练集样本量、协变量维数
sample_size_N, _ = X_unlabeled.shape  # 无标签域样本量

# 缺失值填补：将Y_labeled_train中的NaN值替换为1（对应MATLAB的find(isnan==1)）
Y_labeled_train[np.isnan(Y_labeled_train)] = 1

# 重塑有标签训练数据维度
X_labeled_train = np.reshape(X_labeled_train, (X_labeled_train.shape[0], X_labeled_train.shape[1], 1))

# 重塑有标签训练数据维度
Y_labeled_train = np.reshape(Y_labeled_train, (Y_labeled_train.shape[0], Y_labeled_train.shape[1], 1))

# 构建无标签数据结构（对应MATLAB的struct）
X_unlabeled_struct = {}
X_unlabeled_struct[f"北京"] = [X_unlabeled]


# ======================================== 加载德阳医院数据 ========================================
base_path_deyang = _required_data_path("REALDATA_DEYANG_BASE")
base_path_deyang = base_path_deyang.rstrip("/") + "/"

# 读取德阳医院各类数据
X_labeled_train_deyang = pd.read_csv(f"{base_path_deyang}单任务_X_labeled_train.csv", encoding="gbk").values
X_labeled_test_deyang = pd.read_csv(f"{base_path_deyang}单任务_X_labeled_test.csv", encoding="gbk").values
X_unlabeled_deyang = pd.read_csv(f"{base_path_deyang}单任务_X_unlabeled.csv", encoding="gbk").values

# 合并德阳医院所有数据（对应MATLAB的[]垂直拼接）
X_deyang = np.vstack([X_labeled_train_deyang, X_labeled_test_deyang, X_unlabeled_deyang])

# ======================================== 可选：按北京训练集标准化 ========================================
# MST/MDSP 的均值、协方差距离对特征尺度很敏感；FPCA 得分在不同关节上的量纲可能差异较大。
# 因此实际数据敏感性分析中可以打开 REALDATA_STANDARDIZE=1：
# 只用北京有标签训练集估计均值和标准差，再变换测试集、北京无标签和德阳无标签。
standardize_features = os.environ.get("REALDATA_STANDARDIZE", "0").strip().lower() in {"1", "true", "yes", "y"}
if standardize_features:
    scaler = StandardScaler()
    X_train_2d = X_labeled_train[:, :, 0]
    scaler.fit(X_train_2d)
    X_labeled_train[:, :, 0] = scaler.transform(X_train_2d)
    X_labeled_test = scaler.transform(X_labeled_test)
    X_unlabeled = scaler.transform(X_unlabeled)
    X_deyang = scaler.transform(X_deyang)
    print("已按北京有标签训练集对所有实际数据特征做标准化。")

# ======================================== 可选：统一 L2 正则化 ========================================
# 默认不改变旧结果：监督和 PROPOSED 不加 L2，DRESS/PSS 沿用历史脚本中的 0.01。
# 若设置 REALDATA_L2_LAMBDA，例如 REALDATA_L2_LAMBDA=0.01，则四个方法和选择步骤
# 均使用同一个逻辑损失 + lambda * ||beta||^2，便于做公平的稳定性敏感性分析。
l2_lambda_env = os.environ.get("REALDATA_L2_LAMBDA", "").strip()
if l2_lambda_env:
    realdata_l2_lambda = float(l2_lambda_env)
    fixed_lambda_range = np.array([realdata_l2_lambda], dtype=float)
    benchmark_lambda_hat = np.array([[realdata_l2_lambda]], dtype=float)
    proposed_lambda_hat = np.array([[realdata_l2_lambda]], dtype=float)
else:
    realdata_l2_lambda = None
    fixed_lambda_range = None
    benchmark_lambda_hat = np.array([[0.01]], dtype=float)
    proposed_lambda_hat = None

class_weight_mode = os.environ.get("REALDATA_CLASS_WEIGHT", "none").strip().lower()
if class_weight_mode not in {"none", "balanced"}:
    raise ValueError("REALDATA_CLASS_WEIGHT must be 'none' or 'balanced'")
if class_weight_mode == "balanced":
    print("已启用类别平衡逻辑损失: REALDATA_CLASS_WEIGHT=balanced")

optim_tolerance_env = os.environ.get("REALDATA_OPTIM_TOLERANCE", "").strip()
realdata_optim_tolerance = float(optim_tolerance_env) if optim_tolerance_env else None


def make_model_spec():
    """为每个估计器创建一致的逻辑回归工作模型。"""
    return LogisticModelSpec(class_weight=class_weight_mode)

# 将德阳数据添加到无标签数据结构中
X_unlabeled_struct[f"德阳"] = [X_deyang]

# ======================================== 参数初始化 ========================================
# 初始化参数真值（p+1维全零向量）
beta_star = np.zeros((p + 1, 1))

# 算法参数配置
cv_number = 10  # 交叉验证次数
start_point = 0  # 起始点
end_point = 0.9  # 结束点
num_lambda_mu = 100  # mu参数lambda数量
num_lambda_sigma = 100  # sigma参数lambda数量
lambda_start_mu = 0.001  # mu的lambda起始值
lambda_start_sigma = 0.001  # sigma的lambda起始值
a = 3  # 常数a
c_lambda_1_start = 0.001  # lambda1起始值
c_lambda_2_start = 0.001  # lambda2起始值
k = 1  # 常数k
residual_principle = 1e-3  # 残差阈值
iter_max = 50  # 最大迭代次数
multiple_constant = 1.1  # 倍数常数
lambda_range = 0  # lambda范围标识
num_lambda_1 = 10  # lambda1数量
num_lambda_2 = 10  # lambda2数量
direct_if = 1  # 直接标识
numFolds = 5  # 折数

# ======================================== 调用样本选择函数 ========================================
# 初始化核心类（MstMdsp）
from MstMdsp import MstMdsp              # 半监督选择核心类（之前补全的）
mst_mdsp = MstMdsp(random_seed=123, model_spec=make_model_spec())      # 需确保该类已实现MstMdsp_sample_selection方法

(result, X_labeled, Y_labeled, X_unlabeled, X_unlabeled_combine,
 X_unlabeled_select, select_fields, select_index, all_fields, beta_star) = mst_mdsp.MstMdsp_sample_selection(
    # 有标签数据的协变量[数组](样本量*维度*模拟次数(通常为1))
    X_labeled=X_labeled_train,
    # 无标签数据的协变量[字典](无标签数据集个数个键和值，
    # 键为m1s1,...mksk, 值为列表，每个列表中包含着模拟次数(通常为1)个数组)
    X_unlabeled=X_unlabeled_struct,
    # 有标签数据的因变量[数组](样本量*1*模拟次数(通常为1))
    Y_labeled=Y_labeled_train,
    # 数据生成的真实值,非模拟实验时一般取None,模拟时取beta_star
    beta_star=beta_star,
    # 交叉验证折数,默认取10
    cv_number=None,
    # 测试集划分起点,默认取0
    start_point=None,
    # 测试集划分重点,默认取0.8
    end_point=None,
    # lambda搜索步长倍数,默认取1.1
    multiple_constant=None,
    # mu的lambda搜索数量,默认取10
    num_lambda_mu=None,
    # sigma的lambda搜索数量,默认取10
    num_lambda_sigma=None,
    # lambda_1搜索数量,默认取100
    num_lambda_1=None,
    # lambda_2搜索数量,默认取100
    num_lambda_2=None,
    # mu的lambda起始值,默认取0.001
    lambda_start_mu=None,
    # sigma的lambda起始值,默认取0.001
    lambda_start_sigma=None,
    # ADMM mu惩罚参数起始值,默认取0.001
    c_lambda_1_start=None,
    # ADMM sigma惩罚参数起始值,默认取0.001
    c_lambda_2_start=None,
    # ADMM步长参数,默认取1
    k=None,
    # ADMM阈值参数,默认取3
    a=None,
    # ADMM收敛残差,默认取1e-3
    residual_principle=None,
    # ADMM最大迭代次数,默认取50
    iter_max=None,
    # 直接模式标记(自动/手动选择lambda),默认取1
    direct_if=None,
    # lambda搜索范围,默认取0
    lambda_range=fixed_lambda_range,
    # 交叉验证折数(备用),默认取5
    numFolds=None
)

# 打印最终选择结果（可选，便于分析）
print("\n===== 样本选择结果汇总 =====")
print(f"所有无标签数据集的名称为:\n{all_fields}")
print(f"最终选择的无标签数据集的名称为:\n{select_fields}")
print(f"最终选择的无标签数据集来自第几个数据集:\n{select_index}")
print(f"最终选择的无标签数据维度: {X_unlabeled_select[0].shape}")


# ======================================== 监督学习逻辑回归 ========================================
from MstMdsp import MstMdsp              # 半监督选择核心类（之前补全的）
mst_mdsp = MstMdsp(random_seed=123, model_spec=make_model_spec())      # 需确保该类已实现MstMdsp_sample_selection方法
# 调用监督学习逻辑回归函数, 标准的逻辑回归模型，已集成到筛选函数类mst_mdsp中，不使用无标签数据集
beta_hat_supervised, Evaluate_supervised, _ = mst_mdsp.solve_logistic_regression(
    X_labeled=X_labeled,
    Y_labeled=Y_labeled,
    tolerance=realdata_optim_tolerance,
    max_iter=None,
    initial_value=None,
    beta_star=beta_star,
    CP_if=1,
    lambda_range=fixed_lambda_range,
    numFolds=None
)

# ======================================== c1=0 半监督逻辑回归 ========================================
# DRESSSSLogistic 是历史命名；当前实现是在 ModelSpec.ss_loss_and_grad 中令 c1=0。
# 此处不加筛选，直接纳入所有合并后的无标签数据集。
from DRESSSSLogistic import DRESSSSLogistic
DRESSSSLogistic =  DRESSSSLogistic(random_seed=123, model_spec=make_model_spec())
intercept_from_supervised = os.environ.get(
    "REALDATA_INTERCEPT_FROM_SUPERVISED", "0"
).strip().lower() in {"1", "true", "yes", "y"}
beta_hat_DRESS, Evaluate_semi_supervised_DRESS, select_times_semi_supervised_DRESS\
    = DRESSSSLogistic.dress_ss_logistic_regression(
    X_labeled=X_labeled,
    Y_labeled=Y_labeled,
    X_unlabeled=X_unlabeled_combine,
    tolerance=realdata_optim_tolerance,
    max_iter=None,
    initial_value=None,
    beta_star=beta_star,
    Evaluate_supervised=Evaluate_supervised,
    result_summary=result['result_summary'],
    proposed_if=0,
    best_lambda_hat=benchmark_lambda_hat,
    lambda_range=None,
    numFolds=None,
    h_mu=None,
    h_sigma=None,
    intercept_from_supervised=intercept_from_supervised
)

# # 临时赋值（演示用）
# beta_hat_DRESS = np.zeros((p + 1, 1))
# Evaluate_semi_supervised_DRESS = {"SSE": np.ones((p + 1, 1))}
# select_time_DRESS = 0.0

# ======================================== 合并数据半监督逻辑回归 ========================================
# SSLogistic 当前实现对应 c1=n/(n+N) 的半监督估计；此处不加筛选纳入所有无标签数据集。
from SSLogistic import SSLogistic
SSLogistic = SSLogistic(random_seed=123, model_spec=make_model_spec())
beta_hat_combine, Evaluate_semi_supervised_combine, select_time_combine = SSLogistic.ss_logistic_regression(
    X_labeled=X_labeled,
    Y_labeled=Y_labeled,
    X_unlabeled=X_unlabeled_combine,
    tolerance=realdata_optim_tolerance,
    max_iter=None,
    initial_value=None,
    beta_star=beta_star,
    Evaluate_supervised=Evaluate_supervised,
    result_summary=result['result_summary'],
    proposed_if=0,
    best_lambda_hat=benchmark_lambda_hat,
    lambda_range=None,
    numFolds=None,
    h_mu=None,
    h_sigma=None,
    intercept_from_supervised=intercept_from_supervised
)


# ======================================== 选择数据半监督逻辑回归（提出方法） ========================================
from SSLogistic import SSLogistic
SSLogistic = SSLogistic(random_seed=123, model_spec=make_model_spec())
beta_hat_proposed, Evaluate_semi_supervised_proposed, select_time_proposed = SSLogistic.ss_logistic_regression(
    X_labeled=X_labeled,
    Y_labeled=Y_labeled,
    X_unlabeled=X_unlabeled_select,
    tolerance=realdata_optim_tolerance,
    max_iter=None,
    initial_value=None,
    beta_star=beta_star,
    Evaluate_supervised=Evaluate_supervised,
    result_summary=result['result_summary'],
    proposed_if=1,
    best_lambda_hat=proposed_lambda_hat,
    lambda_range=None,
    numFolds=None,
    h_mu=None,
    h_sigma=None,
    intercept_from_supervised=intercept_from_supervised
)



# ======================================== 计算预测准确率 ========================================
# 定义准确率计算函数（对应MATLAB的curracy_predition）
def predict_probability(beta_hat, X_test):
    """逻辑回归预测概率。"""
    X_test_with_intercept = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
    return 1 / (1 + np.exp(-np.dot(X_test_with_intercept, beta_hat)))


def choose_threshold(beta_hat, X_train, Y_train, mode="fixed"):
    """
    在训练集上选择二分类阈值。

    mode="fixed" 使用传统 0.5；mode="train_balanced" 最大化训练集 balanced accuracy；
    mode="train_f1" 最大化训练集 F1。阈值只用训练集选，不使用测试集信息。
    """
    mode = str(mode).lower()
    if mode in {"fixed", "0.5", "default"}:
        return 0.5

    y_true = np.asarray(Y_train).ravel().astype(int)
    prob = predict_probability(beta_hat, X_train).ravel()
    grid = np.unique(np.r_[0.01, np.linspace(0.02, 0.98, 97), 0.99, prob])

    best_threshold = 0.5
    best_score = -np.inf
    for threshold in grid:
        y_pred = (prob > threshold).astype(int)
        if mode == "train_f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif mode == "train_balanced":
            score = balanced_accuracy_score(y_true, y_pred)
        else:
            raise ValueError("REALDATA_THRESHOLD_MODE must be fixed, train_balanced, or train_f1")
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def curracy_predition(beta_hat, X_test, Y_test, threshold=0.5):
    """
    计算逻辑回归模型在测试集上的二分类预测准确率。
    
    参数
    ----
    beta_hat : np.ndarray, shape (p+1, 1) 或 (p+1,)
        模型参数，第一个元素为截距。
    X_test : np.ndarray, shape (n_test, p)
        测试集协变量，不含截距列。
    Y_test : np.ndarray, shape (n_test, 1) 或 (n_test,)
        测试集真实二分类标签。
    
    返回
    ----
    accuracy : float
        以 0.5 为阈值时的平均分类正确率。
    """
    # 逻辑回归预测
    y_pred_prob = predict_probability(beta_hat, X_test)
    # 二分类预测（阈值0.5）
    y_pred = (y_pred_prob > threshold).astype(int)
    # 计算准确率
    accuracy = np.mean(y_pred == Y_test)
    return accuracy


def prediction_metrics(beta_hat, X_test, Y_test, threshold=0.5):
    """返回 Accuracy、Balanced Accuracy 和 F1，便于比较不同 FPCA 预处理版本。"""
    y_pred_prob = predict_probability(beta_hat, X_test)
    y_pred = (y_pred_prob > threshold).astype(int).ravel()
    y_true = np.asarray(Y_test).ravel().astype(int)
    if len(np.unique(y_true)) == 2:
        roc_auc = float(roc_auc_score(y_true, y_pred_prob.ravel()))
        pr_auc = float(average_precision_score(y_true, y_pred_prob.ravel()))
    else:
        roc_auc = np.nan
        pr_auc = np.nan
    return {
        "threshold": float(threshold),
        "accuracy": float(np.mean(y_pred == y_true)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


# 计算各方法的预测准确率。阈值默认 0.5；可通过 REALDATA_THRESHOLD_MODE 切换为训练集选阈值。
threshold_mode = os.environ.get("REALDATA_THRESHOLD_MODE", "fixed").strip().lower()
method_betas = {
    "SUPERVISED": beta_hat_supervised,
    "PROPOSED": beta_hat_proposed,
    "PSS": beta_hat_combine,
    "DRESS": beta_hat_DRESS,
}
thresholds = {
    method: choose_threshold(beta, X_labeled[:, :, 0], Y_labeled[:, :, 0], threshold_mode)
    for method, beta in method_betas.items()
}

curracy_supervised = curracy_predition(beta_hat_supervised, X_labeled_test, Y_labeled_test, thresholds["SUPERVISED"])
curracy_proposed = curracy_predition(beta_hat_proposed, X_labeled_test, Y_labeled_test, thresholds["PROPOSED"])
curracy_combine = curracy_predition(beta_hat_combine, X_labeled_test, Y_labeled_test, thresholds["PSS"])
curracy_DRESS = curracy_predition(beta_hat_DRESS, X_labeled_test, Y_labeled_test, thresholds["DRESS"])

metrics_summary = pd.DataFrame([
    {"method": method, **prediction_metrics(beta, X_labeled_test, Y_labeled_test, thresholds[method])}
    for method, beta in method_betas.items()
])

# 打印准确率结果
print(f"监督学习准确率: {curracy_supervised:.4f}")
print(f"提出方法准确率: {curracy_proposed:.4f}")
print(f"合并数据准确率: {curracy_combine:.4f}")
print(f"DRESS方法准确率: {curracy_DRESS:.4f}")
print("\n预测指标汇总:")
print(metrics_summary)

# ======================================== 计算p值 ========================================
# 计算各方法的p值（对应MATLAB的normcdf，Python使用scipy.stats.norm.cdf）
p_value_supervised = 2 * (1 - norm.cdf(np.abs(  beta_hat_supervised / np.array(Evaluate_supervised["SSE"]).reshape(-1, 1))))
p_value_DRESS = 2 * (1 - norm.cdf(np.abs(beta_hat_DRESS / np.array(Evaluate_semi_supervised_DRESS["SSE"]).reshape(-1, 1))))
p_value_combine = 2 * (1 - norm.cdf(np.abs(beta_hat_combine / np.array(Evaluate_semi_supervised_combine["SSE"]).reshape(-1, 1))))
p_value_proposed = 2 * (1 - norm.cdf(np.abs(beta_hat_proposed / np.array(Evaluate_semi_supervised_proposed["SSE"]).reshape(-1, 1))  ))

# ======================================== 结果整理 ========================================
# 定义特征名称。PC1 数据会输出 12 个关节名称；PC2/PC3 数据会输出每个关节的各个 FPCA 分量。
name = np.array(["Intercept"] + feature_names).reshape(-1, 1)

# 定义列名
column_name = [
    "name", "beta_hat_supervised", "supervised_p_value",
    "beta_hat_DRESS", "dress_p_value", "beta_hat_combine",
    "combine_p_value", "beta_hat_proposed", "proposed_p_value"
]

# 拼接结果数据（确保维度匹配）
result = np.hstack([
    name[:min(len(name), len(beta_hat_supervised))],
    beta_hat_supervised[:min(len(name), len(beta_hat_supervised))],
    p_value_supervised[:min(len(name), len(p_value_supervised))],
    beta_hat_DRESS[:min(len(name), len(beta_hat_DRESS))],
    p_value_DRESS[:min(len(name), len(p_value_DRESS))],
    beta_hat_combine[:min(len(name), len(beta_hat_combine))],
    p_value_combine[:min(len(name), len(p_value_combine))],
    beta_hat_proposed[:min(len(name), len(beta_hat_proposed))],
    p_value_proposed[:min(len(name), len(p_value_proposed))]
])

# 组合列名和结果
final_result = np.vstack([column_name, result])

# 打印最终结果
print("\n最终结果表:")
print(pd.DataFrame(final_result))

result_dir = RESULT_DIR
result_dir.mkdir(parents=True, exist_ok=True)
metrics_summary.to_csv(result_dir / "prediction_metrics.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(final_result).to_csv(result_dir / "coefficient_pvalues.csv", index=False, header=False, encoding="utf-8-sig")
with open(result_dir / "selected_sources.txt", "w", encoding="utf-8") as f:
    f.write(f"beijing_base={base_path_beijing}\n")
    f.write(f"deyang_base={base_path_deyang}\n")
    f.write(f"standardize_features={standardize_features}\n")
    f.write(f"threshold_mode={threshold_mode}\n")
    f.write(f"realdata_l2_lambda={realdata_l2_lambda}\n")
    f.write(f"class_weight_mode={class_weight_mode}\n")
    f.write(f"intercept_from_supervised={intercept_from_supervised}\n")
    f.write(f"all_fields={all_fields}\n")
    f.write(f"select_fields={select_fields}\n")
    f.write(f"select_index={select_index}\n")
print(f"\n结果已保存到: {result_dir}")
