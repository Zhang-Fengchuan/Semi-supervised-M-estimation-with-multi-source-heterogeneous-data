"""
DataGenerator.py — 半监督 M-估计模拟数据生成器
================================================
本文件负责生成模拟实验用的有标签数据、多个无标签 source，以及用于评估的
`beta_star`。代码中的 X 生成机制和 Y|X 生成机制是分开的：

- X 的有标签域默认来自对称混合高斯 0.5*N(mu, Sigma)+0.5*N(-mu, Sigma)。
  若初始化时设置 x_distribution="single_gaussian"，则改为单一高斯 N(0, Sigma)。
- Y 由 `model_spec.generate_y(X, true_value)` 生成；默认 `LogisticModelSpec`
  会使用三次多项式 logit DGP，但也可以传入其他 `BaseModelSpec` 子类。

当前支持的无标签数据生成机制：

机制 A（原有，which_Exm=1/2/3）：均值/协方差偏移
  - 未标记数据仍从多元正态分布采样，但均值和协方差相对有标签域有偏移 (h_mu, h_sigma)

机制 B（which_Exm=4）：z 矩匹配下的高阶分布差异 + 低阶偏移有害源
  - F1--F4 与有标签域在当前线性正文口径的辅助函数 z=[1,X,X^2,X^3]
    的一二阶矩上总体一致，但高阶尾部分布不同。由于 Cov(z) 最高涉及 X 的
    6 阶矩，这里直接构造 U 的 1--6 阶矩与标准正态完全一致。
  - F1--F4 通过六点对称分布生成，尾部位置 R 和尾部强度 tau 控制 8 阶及以上
    高阶矩差异；默认参数为 (R,tau)=(5,2),(6,2),(6,3),(8,3)。
  - F5--F6 为额外低阶偏移源，均值和协方差都偏离目标域，用来检验 PROPOSED
    是否能筛掉明显有害 source。

注意：文件名和部分函数名沿用了早期“logistic”命名，但当前代码已经通过
`model_spec` 泛化。只有默认不传 `model_spec` 时才是逻辑回归模拟。
"""

import numpy as np
import scipy.stats as stats
from scipy.linalg import eig, sqrtm
from scipy.optimize import minimize
import warnings
import matplotlib.pyplot as plt

from ModelSpec import BaseModelSpec, LogisticModelSpec

warnings.filterwarnings('ignore')


class DataGenerator:
    """
    半监督 M-估计模拟数据生成器。

    `data_generation()` 是主入口。它根据 `which_Exm` 生成：
      - `X_labeled`, `Y_labeled`: 有标签样本，形状为 (n, p, T) 和 (n, 1, T)；
      - `X_unlabeled`: 多 source 无标签样本字典；
      - `beta_star`: 用大样本监督 M-估计近似得到的目标参数。

    这里的 `beta_star` 不是 `true_value` 的简单截取。默认 LogisticModelSpec 的 DGP
    含高次项，而估计模型只用线性 logit，因此 `beta_star` 是“估计模型在有标签总体分布下的
    伪真值”，通过 100000 个样本数值优化近似得到。
    """

    def __init__(self, random_seed=123, model_spec=None, x_distribution="mixture_gaussian"):
        """
        初始化数据生成器，设置随机种子和模型规范。

        参数
        ----
        random_seed : int，可选
            全局随机种子，确保模拟结果可复现，默认 123。
        model_spec  : BaseModelSpec 子类实例，可选
            M-估计模型规范对象，控制 Y 的生成方式和参数估计目标函数。
            默认为 None，此时自动使用 LogisticModelSpec()（逻辑回归）。
        x_distribution : {"mixture_gaussian", "single_gaussian"}，可选
            目标域 X 的生成机制。默认保持论文逻辑回归实验的对称混合高斯；
            线性回归实验可设为 single_gaussian，令目标域 X ~ N(0, Sigma)。
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)   # 设置全局随机种子，确保每次运行结果一致
        self.beta_star = None         # 真实参数（由大样本数值估计得到）
        # 若未指定模型规范，默认使用逻辑回归模型
        self.model_spec = model_spec if model_spec is not None else LogisticModelSpec()
        if x_distribution not in {"mixture_gaussian", "single_gaussian"}:
            raise ValueError("x_distribution must be 'mixture_gaussian' or 'single_gaussian'.")
        self.x_distribution = x_distribution

    def _sample_x(self, sample_size, p, mu, Sigma, shift_if=0, h_mu=0.0, h_sigma=0.0):
        """
        生成一次模拟的 X。

        mixture_gaussian:
            X ~ 0.5 N(mu + h_mu 1, Sigma + h_sigma I)
              + 0.5 N(-mu + h_mu 1, Sigma + h_sigma I).
        single_gaussian:
            X ~ N(h_mu 1, Sigma + h_sigma I)，目标域 h_mu=h_sigma=0。
        """
        sample_size = int(sample_size)
        p = int(p)
        shift_if = 0 if shift_if is None else shift_if
        h_mu = 0.0 if h_mu is None else float(h_mu)
        h_sigma = 0.0 if h_sigma is None else float(h_sigma)
        Sigma_t = Sigma + (h_sigma if shift_if == 1 else 0.0) * np.eye(p)
        Sigma_t = 0.5 * (Sigma_t + Sigma_t.T)

        if self.x_distribution == "single_gaussian":
            mean = (h_mu if shift_if == 1 else 0.0) * np.ones(p)
            return stats.multivariate_normal.rvs(
                mean=mean, cov=Sigma_t, size=sample_size
            ).reshape(sample_size, p)

        n1 = int(np.floor(0.5 * sample_size))
        n2 = sample_size - n1
        mu_shift = (h_mu if shift_if == 1 else 0.0) * np.ones((p, 1))
        X1 = stats.multivariate_normal.rvs(
            mean=(mu + mu_shift).ravel(), cov=Sigma_t, size=n1
        ).reshape(n1, p)
        X2 = stats.multivariate_normal.rvs(
            mean=(-mu + mu_shift).ravel(), cov=Sigma_t, size=n2
        ).reshape(n2, p)
        X = np.vstack([X1, X2])
        return X[np.random.permutation(sample_size), :]

    # ==========================================================================
    # 1. 基础 X/Y 生成函数
    # ==========================================================================

    def data_generation_labeled(self, sample_size=None, p=None, simulation_size=None, true_value=None,
                                mu=None, Sigma=None, shift_if=None, h_mu=None, h_sigma=None,
                                unlabeled_if=None, show_plot=False):
        """
        生成有标签或无标签数据（X 分布机制不变，Y 生成已泛化至任意 model_spec）。

        默认 X 服从混合高斯分布：0.5·N(μ, Σ) + 0.5·N(-μ, Σ)。
        若 x_distribution="single_gaussian"，则 X ~ N(0, Σ)；当 shift_if=1 时，
        单一高斯 source 变为 N(h_mu·1, Σ+h_sigma·I)。
        Y 由 self.model_spec.generate_y(X, true_value) 生成。

        参数
        ----
        sample_size    : int，可选
            每次模拟的样本量，默认 250。
        p              : int，可选
            特征维度，默认 5。
        simulation_size: int，可选
            独立模拟次数 T，默认 1000。
        true_value     : np.ndarray，可选
            DGP 真实参数 [d_dgp, 1]。None 时由 model_spec.default_true_value(p) 决定。
        mu             : np.ndarray，可选
            混合高斯分布的分量均值 [p, 1]，默认为 0.5·ones(p)。
        Sigma          : np.ndarray，可选
            混合高斯分布的协方差矩阵 [p, p]，默认为 AR(1) 结构 0.8^|i-j|。
        shift_if       : int，可选
            0（默认）= 无分布偏移；1 = 有均值+协方差偏移（用于无标签数据）。
        h_mu           : float，可选
            shift_if=1 时生效，均值偏移量（所有维度等量平移）。
        h_sigma        : float，可选
            shift_if=1 时生效，协方差偏移量（在对角线加 h_sigma·I）。
        unlabeled_if   : int，可选
            1（默认）= 仅生成 X（不生成 Y，模拟无标签数据）；
            0 = 同时生成 X 和 Y（有标签数据）。
        show_plot      : bool，可选
            True = 在最后一次模拟时绘制 X1 vs X2 散点图（仅 unlabeled_if=0 时有效）。

        返回
        ----
        X_labeled : np.ndarray，形状 [sample_size, p, simulation_size]
            生成的特征矩阵（三维数组，最后一维为模拟轮次）。
        Y_labeled : np.ndarray，形状 [sample_size, 1, simulation_size]
            生成的响应变量（unlabeled_if=1 时为全零占位数组）。
        """
        # ---- 参数默认值处理 ----
        sample_size = 250 if sample_size is None else sample_size
        p = 5 if p is None else p
        # true_value 默认值由 model_spec 决定（不再硬编码逻辑回归参数）
        if true_value is None:
            true_value = self.model_spec.default_true_value(p)
        simulation_size = 1000 if simulation_size is None else simulation_size
        # 默认均值：混合高斯分量均值为 0.5·1_p（非零均值保证 X 的分布不退化到均值为 0）
        mu = 0.5 * np.ones((p, 1)) if mu is None else mu
        if Sigma is None:
            # AR(1) 协方差结构：相关系数随距离指数衰减，Σ_{ij} = 0.8^|i-j|
            Sigma = np.array([[0.8 ** abs(i - j) for j in range(p)] for i in range(p)])
        shift_if = 0 if shift_if is None else shift_if
        unlabeled_if = 1 if unlabeled_if is None else unlabeled_if

        # ---- 生成特征 X ----
        X_labeled = np.zeros((sample_size, p, simulation_size))
        for t in range(simulation_size):
            X_labeled[:, :, t] = self._sample_x(
                sample_size=sample_size,
                p=p,
                mu=mu,
                Sigma=Sigma,
                shift_if=shift_if,
                h_mu=h_mu,
                h_sigma=h_sigma,
            )

        # ---- 生成响应变量 Y（已泛化：委托给 self.model_spec.generate_y）----
        Y_labeled = np.zeros((sample_size, 1, simulation_size))  # 全零占位（无标签时不生成 Y）

        if unlabeled_if != 1:
            # unlabeled_if=0 时才生成 Y（有标签数据）
            for t in range(simulation_size):
                Y_labeled[:, :, t] = self.model_spec.generate_y(
                    X_labeled[:, :, t], true_value
                )
                # 最后一次模拟时可选绘制散点图（用于可视化检验 DGP）
                if show_plot and (t == simulation_size - 1):
                    self._plot_labeled(X_labeled[:, :, t], Y_labeled[:, 0, t])

        return X_labeled, Y_labeled

    # ==========================================================================
    # 2. 高阶分布 source 单次生成
    # ==========================================================================

    def _compute_total_moments(self, mu0, Sigma0):
        """
        计算目标 X 分布的总体均值和协方差。

        mixture_gaussian 时由混合高斯的矩公式可推导：
          - 一阶矩（均值）：μ_T = 0.5·μ_0 + 0.5·(-μ_0) = 0
          - 二阶矩（协方差）：V_T = Σ_0 + μ_0·μ_0^T
            （混合协方差 = 各分量协方差 + 各分量均值的外积之和乘以混合权重）
        single_gaussian 时目标域为 N(0, Σ_0)，所以 μ_T=0, V_T=Σ_0。

        该结果用于 Exm4：F1–F4 的 X 通过线性变换 X = μ_T + L_T·U 生成，
        其中 L_T 使得 Cov[X] = V_T；F5–F6 在此基础上再加入低阶偏移。

        参数
        ----
        mu0    : np.ndarray，形状 [p, 1]
            有标签域混合高斯分量均值向量。
        Sigma0 : np.ndarray，形状 [p, p]
            有标签域混合高斯分量协方差矩阵。

        返回
        ----
        mu_T : np.ndarray，形状 [p, 1]
            总体均值（理论上为零向量）。
        V_T  : np.ndarray，形状 [p, p]
            总体协方差矩阵。
        L_T  : np.ndarray，形状 [p, p]
            V_T 的 Cholesky 下三角因子，满足 L_T @ L_T.T = V_T。
        """
        mu_T = np.zeros_like(mu0)           # 两种默认目标域都以 0 为总体均值
        if self.x_distribution == "single_gaussian":
            V_T = Sigma0.copy()
        else:
            V_T = Sigma0 + mu0 @ mu0.T      # Σ_0 + μ_0·μ_0^T（与混合高斯推导一致）
        V_T = (V_T + V_T.T) / 2            # 强制对称，避免数值误差导致非对称

        # 尝试 Cholesky 分解（要求 V_T 正定）；若失败则退化到矩阵平方根
        try:
            L_T = np.linalg.cholesky(V_T)  # Cholesky 分解速度更快且数值更稳定
        except np.linalg.LinAlgError:
            # V_T 数值上不完全正定时，用矩阵平方根作为备用方案
            L_T = np.real(sqrtm(V_T))
        return mu_T, V_T, L_T

    def _sample_gauss_hermite_standard_normal(self, sample_size, p, n_nodes):
        """
        用 m 点 Gauss-Hermite 求积构造一个离散随机变量 U。

        若 hermgauss(m) 给出节点 x_j 和权重 w_j，则
            u_j = sqrt(2) x_j,     pi_j = w_j / sqrt(pi)
        构成的离散分布对标准正态的多项式矩积分精确到 2m-1 阶。

        对当前线性正文实验，z=[1,X,X^2,X^3]，Cov(z) 最多涉及 X 的 6 阶矩。
        因此 m>=4 时，U 的 0--6 阶矩与 N(0,1) 完全一致，而更高阶矩不同。
        """
        n_nodes = int(n_nodes)
        nodes, weights = np.polynomial.hermite.hermgauss(n_nodes)
        nodes = np.sqrt(2.0) * nodes
        probabilities = weights / np.sqrt(np.pi)
        probabilities = probabilities / probabilities.sum()
        index = np.random.choice(n_nodes, size=(int(sample_size), int(p)), p=probabilities)
        return nodes[index]

    def _six_point_moment_matched_parameters(self, R, tau):
        """
        计算六点对称分布的支撑点和概率。

        目标是构造 U，使其与标准正态共享 1--6 阶矩：
            E(U)=E(U^3)=E(U^5)=0, E(U^2)=1, E(U^4)=3, E(U^6)=15。
        分布在尾部点 ±R 上放总概率 q=tau/R^6，其余概率放在中心点
        ±sqrt(y1), ±sqrt(y2) 上并自动补偿低阶矩。

        R 越大、tau 越大，8 阶及以上尾部矩差异越强。
        """
        R = float(R)
        tau = float(tau)
        if R <= 0 or tau <= 0:
            raise ValueError("R and tau must be positive for six-point moment matching.")

        q = tau / (R ** 6)
        if not (0.0 < q < 1.0):
            raise ValueError(f"Invalid six-point parameters: q={q:.6g}, need 0<q<1.")

        m2 = (1.0 - q * R ** 2) / (1.0 - q)
        m4 = (3.0 - q * R ** 4) / (1.0 - q)
        m6 = (15.0 - q * R ** 6) / (1.0 - q)
        denom = m4 - m2 ** 2
        if denom <= 0:
            raise ValueError("Invalid six-point parameters: central fourth moment is degenerate.")

        s = (m6 - m2 * m4) / denom
        c = s * m2 - m4
        disc = s ** 2 - 4.0 * c
        if disc < 0:
            raise ValueError("Invalid six-point parameters: central support is not real.")

        y1 = (s - np.sqrt(disc)) / 2.0
        y2 = (s + np.sqrt(disc)) / 2.0
        if y1 <= 0 or y2 <= 0 or np.isclose(y1, y2):
            raise ValueError("Invalid six-point parameters: central support is not positive.")

        rho = (y2 - m2) / (y2 - y1)
        if not (-1e-10 <= rho <= 1.0 + 1e-10):
            raise ValueError(f"Invalid six-point parameters: rho={rho:.6g} not in [0,1].")
        rho = min(1.0, max(0.0, rho))

        a = np.sqrt(y1)
        b = np.sqrt(y2)
        support = np.array([-R, -b, -a, a, b, R], dtype=float)
        probabilities = np.array([
            q / 2.0,
            (1.0 - q) * (1.0 - rho) / 2.0,
            (1.0 - q) * rho / 2.0,
            (1.0 - q) * rho / 2.0,
            (1.0 - q) * (1.0 - rho) / 2.0,
            q / 2.0,
        ], dtype=float)
        probabilities = probabilities / probabilities.sum()
        return support, probabilities

    def _sample_six_point_moment_matched(self, sample_size, p, R, tau):
        """按六点矩匹配分布生成独立分量 U。"""
        support, probabilities = self._six_point_moment_matched_parameters(R, tau)
        index = np.random.choice(len(support), size=(int(sample_size), int(p)), p=probabilities)
        return support[index]

    def data_generation_unlabeled_single_source(self, source_type, sample_size, p,
                                                 mu_T, L_T, mu0=None, Sigma0=None,
                                                 c5_contamination_ratio=0.05,
                                                 q6_mu_shift=0.5, q6_sigma_shift=0.5):
        """
        生成单个高阶分布 source 的一次模拟样本 X^(k)。

        核心思想：通过线性变换 X = μ_T + L_T · U 将标准化随机向量 U 映射到
        目标分布空间，使得 E[X] = μ_T，Cov[X] = V_T（当 E[U]=0, Cov[U]=I_p 时）。
        不同 source 对应 U 的不同高阶分布，从而在保持一、二阶矩相同的同时引入分布差异。

        参数
        ----
        source_type            : str
            source 类型。正文 Example 4 使用 'F1'~'F6'（不区分大小写）。
            F1--F4 使用六点矩匹配分布，使 z 的一二阶矩与目标域一致，但高阶矩不同；
            F5--F6 是低阶均值/协方差偏移源，用于检验筛选能力。
            'Q1'~'Q6' 是早期调试口径的兼容别名。
        sample_size            : int
            该 source 的单次模拟样本量 N。
        p                      : int
            特征维度。
        mu_T                   : np.ndarray，形状 [p, 1]
            总体均值（通常为零向量）。
        L_T                    : np.ndarray，形状 [p, p]
            V_T 的 Cholesky 因子，满足 L_T @ L_T.T = V_T。
        mu0                    : np.ndarray，可选，形状 [p, 1]
            混合高斯分量均值（Q1 使用），默认 None。
        Sigma0                 : np.ndarray，可选，形状 [p, p]
            混合高斯分量协方差（Q1 使用），默认 None。
        c5_contamination_ratio : float，可选
            Q5 旧污染正态的污染比例 ε，默认 0.05。仅保留给旧调试口径使用。
        q6_mu_shift            : float，可选
            Q6 额外均值偏移量 δ_μ（加到 μ_T 上），默认 0.5。
        q6_sigma_shift         : float，可选
            Q6 额外协方差偏移量 δ_Σ（加到 V_T 对角线上），默认 0.5。

        返回
        ----
        X : np.ndarray，形状 [sample_size, p]
            一次模拟生成的无标签 source 样本矩阵。
        """
        p = int(p)
        source_type = source_type.upper()  # 统一转为大写，避免大小写问题

        if source_type == 'Q1':
            # ---- Q1：与有标签域完全相同的 X 分布 ----
            # 用于验证：当 source 与有标签域分布完全一致时，方法应充分利用该 source
            assert mu0 is not None and Sigma0 is not None, "Q1 需要提供 mu0 和 Sigma0"
            X = self._sample_x(
                sample_size=sample_size,
                p=p,
                mu=mu0,
                Sigma=Sigma0,
                shift_if=0,
            )

        elif source_type == 'Q2':
            # ---- Q2：各维度独立标准正态 U_j ~ N(0,1) ----
            # E[U]=0, Cov[U]=I_p，满足矩条件
            # 经变换后 X = L_T·U 满足 E[X]=μ_T, Cov[X]=L_T·I_p·L_T^T = V_T
            U = np.random.randn(sample_size, p)            # [sample_size, p]，i.i.d. N(0,1)
            X = mu_T.T + (L_T @ U.T).T                    # 线性变换到目标矩空间

        elif source_type in {'F1', 'F2', 'F3', 'F4'}:
            # ---- F1--F4：六点矩匹配分布 ----
            # 四个 source 都精确匹配标准正态的 1--6 阶矩，因此当前 z 的一二阶矩一致。
            # R 和 tau 从小到大增强 8 阶及以上高阶尾部差异。
            params = {
                'F1': (5.0, 2.0),
                'F2': (6.0, 2.0),
                'F3': (6.0, 3.0),
                'F4': (8.0, 3.0),
            }
            R, tau = params[source_type]
            U = self._sample_six_point_moment_matched(sample_size, p, R=R, tau=tau)
            X = mu_T.T + (L_T @ U.T).T

        elif source_type == 'Q3':
            # ---- Q3：旧调试口径，U_j = sqrt(3/5) * T_j, T_j ~ t_5 ----
            scale = np.sqrt(3.0 / 5.0)
            U = scale * stats.t.rvs(df=5, size=(sample_size, p))
            X = mu_T.T + (L_T @ U.T).T

        elif source_type == 'Q4':
            # ---- Q4：旧调试口径，强右偏标准化对数正态 W~LN(0,1) ----
            mu_ln = np.exp(0.5)
            std_ln = np.sqrt((np.e - 1) * np.e)
            W = np.random.lognormal(mean=0.0, sigma=1.0, size=(sample_size, p))
            U = (W - mu_ln) / std_ln
            X = mu_T.T + (L_T @ U.T).T

        elif source_type == 'Q5':
            # ---- Q5：旧调试口径，污染正态 (1-ε)·N(0,c1)+ε·N(0,9)，总方差=1 ----
            eps = c5_contamination_ratio
            c1 = (1.0 - 9.0 * eps) / (1.0 - eps)
            if c1 <= 0:
                raise ValueError(
                    f"{source_type}: c5_contamination_ratio={eps} 过大，导致 c1={c1:.4f}≤0。"
                    f"建议 eps < 1/9 ≈ 0.111。"
                )
            mask = np.random.rand(sample_size, p) < eps
            U = np.where(
                mask,
                np.random.randn(sample_size, p) * 3.0,
                np.random.randn(sample_size, p) * np.sqrt(c1)
            )
            X = mu_T.T + (L_T @ U.T).T

        elif source_type in {'F5', 'F6'}:
            # ---- F5--F6：低阶偏移有害源 ----
            # 均值分别向 -1_p / +1_p 平移，协方差增加 I_p，故 z 的低阶矩也明显不同。
            V_T = L_T @ L_T.T
            Sigma_shift = 0.5 * (V_T + V_T.T) + np.eye(p)
            sign = -1.0 if source_type == 'F5' else 1.0
            mean_shift = mu_T.ravel() + sign * np.ones(p)
            X = stats.multivariate_normal.rvs(
                mean=mean_shift, cov=Sigma_shift, size=sample_size
            ).reshape(sample_size, p)

        elif source_type == 'Q6':
            # ---- Q6：低阶偏移（均值+协方差均与有标签域不同）----
            # Q6 有意违反"一二阶矩相同"的假设，用于测试方法能否识别并拒绝有害 source
            # 分布：N(μ_T + δ_μ·1_p, V_T + δ_Σ·I_p)
            from scipy.linalg import cholesky
            V_T = L_T @ L_T.T                              # 从 Cholesky 因子还原 V_T
            Sigma_q6 = V_T + q6_sigma_shift * np.eye(p)   # 协方差膨胀
            Sigma_q6 = (Sigma_q6 + Sigma_q6.T) / 2        # 强制对称
            mu_q6 = mu_T + q6_mu_shift * np.ones((p, 1))  # 均值偏移
            X = stats.multivariate_normal.rvs(
                mean=mu_q6.ravel(), cov=Sigma_q6, size=sample_size
            ).reshape(sample_size, p)

        else:
            raise ValueError(
                f"未知的 source_type='{source_type}'，正文 Exm4 可选 F1/F2/F3/F4/F5/F6；"
                "兼容旧口径可选 Q1/Q2/Q3/Q4/Q5/Q6。"
            )

        return X  # [sample_size, p]

    # ==========================================================================
    # 3. 主函数 data_generation（扩展，向后兼容）
    # ==========================================================================

    def data_generation(self, which_Exm, sample_size_n=None, sample_size_N=None, p=None,
                        true_value=None, simulation_times=None,
                        higher_order_sources=None, q6_mu_shift=0.5, q6_sigma_shift=0.5,
                        beta_star_tolerance=None, beta_star_max_iter=None,
                        source_h_mu=None, source_h_sigma=None):
        """
        主数据生成函数（扩展版，向后兼容原有接口）。

        根据 which_Exm 选择实验类型，生成有标签数据、无标签数据（多 source）
        以及数值估计的真实参数 beta_star。

        参数
        ----
        which_Exm         : int
            实验类型编号：
            1 → 均值/协方差偏移，对角组合（|h_mu|=|h_sigma|），5 维特征
            2 → 均值/协方差偏移，全组合（h_mu × h_sigma 笛卡尔积），5 维特征
            3 → 均值/协方差偏移，对角组合，50 维特征（高维实验）
            4 → z 矩匹配下的高阶分布差异（F1–F4）+ 低阶偏移源（F5–F6），5 维特征
            5 → 混合实验（部分偏移 source + 部分高阶分布 source）
        sample_size_n     : int，可选
            有标签数据样本量 n，默认 250。
        sample_size_N     : int，可选
            每个无标签 source 的样本量 N，默认 5000（对所有实验类型）。
        p                 : int，可选
            特征维度，默认值随 which_Exm 变化（1/2/4/5 默认 5，3 默认 50）。
        true_value        : np.ndarray，可选
            DGP 真实参数，None 时由 model_spec.default_true_value(p) 提供。
        simulation_times  : int，可选
            独立模拟次数 T，默认 100。
        higher_order_sources : list of str，可选
            which_Exm=4/5 时使用的高阶分布 source 列表。
            which_Exm=4 默认 ['F1','F2','F3','F4','F5','F6']，对应正文 Example 4。
            which_Exm=5 是旧调试混合场景，默认 ['Q2','Q3','Q4','Q5','Q6']。
        q6_mu_shift       : float，可选
            Q6 source 的均值偏移量，默认 0.5。
        q6_sigma_shift    : float，可选
            Q6 source 的协方差对角偏移量，默认 0.5。
        beta_star_tolerance : float，可选
            计算 beta_star 时传给 BFGS 的梯度收敛容差。None 时默认 1e-6。
        beta_star_max_iter : int，可选
            计算 beta_star 时传给 BFGS 的最大迭代次数。None 时默认 1000。
        source_h_mu, source_h_sigma : list of float, 可选
            覆盖 Exm1/2/3 默认 source 均值/协方差偏移列表。None 时保持论文默认设置。

        返回
        ----
        X_labeled    : np.ndarray，形状 [n, p, T]
            有标签数据特征（n 个样本，p 维，T 次模拟）。
        Y_labeled    : np.ndarray，形状 [n, 1, T]
            有标签数据响应变量（T 次模拟）。
        X_unlabeled  : dict
            无标签数据字典，键为 source 名称（如 'm1s1'、'F1'），
            值为长度 T 的列表，每个元素为 [N, p] 的样本矩阵。
        beta_star    : np.ndarray，形状 [p+1, 1]
            用 100000 样本数值估计的"真实"参数（作为评估基准）。
        fields_X     : list of str
            X_unlabeled 的所有键名列表。
        h_mu         : list
            均值偏移量列表（Exm1/2/3）或 source 名称列表（Exm4/5）。
        h_sigma      : list
            协方差偏移量列表（Exm1/2/3）或 source 名称列表（Exm4/5）。
        """
        # ---- 参数默认值处理 ----
        sample_size_n = 250 if sample_size_n is None else sample_size_n
        simulation_times = 100 if simulation_times is None else simulation_times
        beta_star_tolerance = 1e-6 if beta_star_tolerance is None else beta_star_tolerance
        beta_star_max_iter = 1000 if beta_star_max_iter is None else beta_star_max_iter

        def _maybe_override_shift(default_mu, default_sigma):
            mu_list = default_mu if source_h_mu is None else list(source_h_mu)
            sigma_list = default_sigma if source_h_sigma is None else list(source_h_sigma)
            if len(mu_list) == 0 or len(sigma_list) == 0:
                raise ValueError("source_h_mu/source_h_sigma must be non-empty when provided.")
            return mu_list, sigma_list

        # ---- 根据实验类型配置偏移参数和生成模式 ----
        if which_Exm == 1:
            # 实验1：5维特征，对角偏移（均值偏移量=协方差偏移量）
            p = 5 if p is None else p
            sample_size_N = sample_size_N if sample_size_N is not None else 5000
            h_mu, h_sigma = _maybe_override_shift(
                [-0.01, 0.01, -0.5, 0.5, -1.0, 1.0],
                [ 0.01, 0.01,  0.5, 0.5,  1.0, 1.0],
            )
            if len(h_mu) != len(h_sigma):
                raise ValueError("Exm1 diagonal sources require source_h_mu and source_h_sigma to have the same length.")
            unlabeled_mode = 'shift_diagonal'   # 对角组合：source_i 用 (h_mu[i], h_sigma[i])
        elif which_Exm == 2:
            # 实验2：5维特征，全组合偏移（h_mu × h_sigma 笛卡尔积，共 36 个 source）
            p = 5 if p is None else p
            sample_size_N = sample_size_N if sample_size_N is not None else 5000
            h_mu, h_sigma = _maybe_override_shift(
                [-0.01, 0.01, -0.5, 0.5, -1.0, 1.0],
                [ 0.01, 0.01,  0.5, 0.5,  1.0, 1.0],
            )
            unlabeled_mode = 'shift_full'        # 全组合：每对 (h_mu[i], h_sigma[j]) 一个 source
        elif which_Exm == 3:
            # 实验3：50维高维特征，对角偏移（测试高维情形）
            p = 50 if p is None else p
            sample_size_N = sample_size_N if sample_size_N is not None else 5000
            h_mu, h_sigma = _maybe_override_shift([0.01, 0.5, 1.0], [0.01, 0.5, 1.0])
            if len(h_mu) != len(h_sigma):
                raise ValueError("Exm3 diagonal sources require source_h_mu and source_h_sigma to have the same length.")
            unlabeled_mode = 'shift_diagonal'
        elif which_Exm == 4:
            # 实验4：四个高阶异质 source + 两个低阶偏移有害 source。
            p = 5 if p is None else p
            sample_size_N = sample_size_N if sample_size_N is not None else 5000
            if higher_order_sources is None:
                higher_order_sources = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6']
            h_mu    = higher_order_sources      # 用 source 名称代替偏移量（描述性占位）
            h_sigma = higher_order_sources
            unlabeled_mode = 'higher_order'
        elif which_Exm == 5:
            # 实验5：混合实验（同时含偏移 source 和高阶分布 source）
            p = 5 if p is None else p
            sample_size_N = sample_size_N if sample_size_N is not None else 5000
            if higher_order_sources is None:
                higher_order_sources = ['Q2', 'Q3', 'Q4', 'Q5', 'Q6']
            h_mu    = [-0.5, 0.5, -1.0, 1.0] + higher_order_sources  # 前四个偏移 source
            h_sigma = [ 0.5, 0.5,  1.0, 1.0] + higher_order_sources
            unlabeled_mode = 'mixed'
        else:
            # 未识别的实验编号：退化为实验1的配置
            p = 5 if p is None else p
            sample_size_N = sample_size_N if sample_size_N is not None else 5000
            h_mu    = [-0.01, 0.01, -0.5, 0.5, -1.0, 1.0]
            h_sigma = [ 0.01, 0.01,  0.5, 0.5,  1.0, 1.0]
            unlabeled_mode = 'shift_diagonal'

        # ---- 真实参数（由 model_spec.default_true_value 提供模型无关的默认值）----
        if true_value is None:
            true_value = self.model_spec.default_true_value(p)

        # ---- 基础分布参数（有标签域的 X 分布参数）----
        np.random.seed(self.random_seed)    # 重置全局种子，确保每次调用结果可复现
        mu0    = 0.5 * np.ones((p, 1))     # 混合高斯分量均值；single_gaussian 下不作为均值使用
        Sigma0 = np.array([[0.8 ** abs(i - j) for j in range(p)] for i in range(p)])
        # AR(1) 协方差结构，相关系数以 0.8 为底随维度距离衰减

        # ---- 计算总体均值/协方差/Cholesky（高阶矩实验的变换基础）----
        # 仅高阶矩实验（Exm4/5）用到，但所有情况下均预先计算以统一流程
        mu_T, V_T, L_T = self._compute_total_moments(mu0, Sigma0)

        # 用大样本估计 beta_star。默认逻辑回归 DGP 含高次项而估计模型只含线性项，
        # 因此这里得到的是估计模型的总体风险最小点（伪真值），不是 true_value 本身。
        # 求解委托给 model_spec.solve_supervised：逻辑回归走 BFGS，线性回归走闭式 OLS。
        X_large, Y_large = self.data_generation_labeled(
            sample_size=100000, p=p, simulation_size=1,
            true_value=true_value, mu=mu0, Sigma=Sigma0,
            shift_if=0, unlabeled_if=0
        )
        beta_star = self.model_spec.solve_supervised(
            X_large[:, :, 0],
            Y_large[:, :, 0],
            lambda_reg=0.0,
            initial_value=None,
            tolerance=beta_star_tolerance,
            max_iter=beta_star_max_iter,
        )
        beta_star = beta_star.reshape(-1, 1)   # 整形为列向量 [p+1, 1]

        # ---- 生成有标签数据（T 次模拟，每次 n 个样本）----
        # 重置为同一个种子是为了复现实验，而不是为了让三批数据统计独立。
        # beta_star 大样本、有标签样本、无标签样本的随机数流会从相同种子重新开始。
        np.random.seed(self.random_seed)
        X_labeled, Y_labeled = self.data_generation_labeled(
            sample_size=sample_size_n, p=p, simulation_size=simulation_times,
            true_value=true_value, mu=mu0, Sigma=Sigma0,
            shift_if=0, unlabeled_if=0
        )

        # ---- 生成无标签数据（各 source 的 T 次模拟数据）----
        # 再次重置随机种子，保证相同参数下无标签数据可完全复现。
        X_unlabeled = {}                    # 字典：{source_key: [T个[N,p]数组]}

        if unlabeled_mode == 'shift_diagonal':
            # ---- 原机制 A 对角版：第 i 个 source 使用 (h_mu[i], h_sigma[i]) ----
            # 对角组合保证偏移量大小一致，适合对称分析
            for i in range(len(h_mu)):
                X_mid, _ = self.data_generation_labeled(
                    sample_size=sample_size_N, p=p, simulation_size=simulation_times,
                    true_value=true_value, mu=mu0, Sigma=Sigma0,
                    shift_if=1, h_mu=h_mu[i], h_sigma=h_sigma[i], unlabeled_if=1
                )
                key = f'm{i+1}s{i+1}'    # source 键名格式：m<均值偏移编号>s<协方差偏移编号>
                X_unlabeled[key] = [X_mid[:, :, k] for k in range(simulation_times)]

        elif unlabeled_mode == 'shift_full':
            # ---- 原机制 A 全组合版：所有 (h_mu[i], h_sigma[j]) 对，共 len(h_mu)×len(h_sigma) 个 source ----
            for i in range(len(h_mu)):
                for j in range(len(h_sigma)):
                    X_mid, _ = self.data_generation_labeled(
                        sample_size=sample_size_N, p=p, simulation_size=simulation_times,
                        true_value=true_value, mu=mu0, Sigma=Sigma0,
                        shift_if=1, h_mu=h_mu[i], h_sigma=h_sigma[j], unlabeled_if=1
                    )
                    key = f'm{i+1}s{j+1}'  # 行索引均值偏移，列索引协方差偏移
                    X_unlabeled[key] = [X_mid[:, :, k] for k in range(simulation_times)]

        elif unlabeled_mode == 'higher_order':
            # ---- 机制 B：正文 Exm4 默认 F1–F6，每个 source 一次模拟循环 ----
            for src in higher_order_sources:
                X_src_list = []
                for t in range(simulation_times):
                    # 每次模拟独立生成一个 [N, p] 样本矩阵
                    X_one = self.data_generation_unlabeled_single_source(
                        source_type=src,
                        sample_size=sample_size_N,
                        p=p,
                        mu_T=mu_T,
                        L_T=L_T,
                        mu0=mu0,
                        Sigma0=Sigma0,
                        q6_mu_shift=q6_mu_shift,
                        q6_sigma_shift=q6_sigma_shift
                    )
                    X_src_list.append(X_one)
                X_unlabeled[src] = X_src_list   # 键名即 source 类型字符串（如 'F1'）

        elif unlabeled_mode == 'mixed':
            # ---- 混合模式：前几个为偏移 source（Exm1 格式），后几个为高阶分布 source ----
            shift_mus    = [-0.5, 0.5, -1.0, 1.0]   # 偏移 source 的均值偏移量
            shift_sigmas = [ 0.5, 0.5,  1.0, 1.0]   # 对应协方差偏移量
            # 生成偏移 source
            for i, (hm, hs) in enumerate(zip(shift_mus, shift_sigmas)):
                X_mid, _ = self.data_generation_labeled(
                    sample_size=sample_size_N, p=p, simulation_size=simulation_times,
                    true_value=true_value, mu=mu0, Sigma=Sigma0,
                    shift_if=1, h_mu=hm, h_sigma=hs, unlabeled_if=1
                )
                key = f'shift_m{i+1}'              # 偏移 source 键名格式：shift_m<编号>
                X_unlabeled[key] = [X_mid[:, :, k] for k in range(simulation_times)]
            # 生成高阶分布 source（与 higher_order 模式相同）
            for src in higher_order_sources:
                X_src_list = []
                for t in range(simulation_times):
                    X_one = self.data_generation_unlabeled_single_source(
                        source_type=src, sample_size=sample_size_N, p=p,
                        mu_T=mu_T, L_T=L_T, mu0=mu0, Sigma0=Sigma0,
                        q6_mu_shift=q6_mu_shift, q6_sigma_shift=q6_sigma_shift
                    )
                    X_src_list.append(X_one)
                X_unlabeled[src] = X_src_list

        fields_X = list(X_unlabeled.keys())  # 所有 source 的键名列表（有序）
        return X_labeled, Y_labeled, X_unlabeled, beta_star, fields_X, h_mu, h_sigma

    # ==========================================================================
    # 4. 监督 M-估计求解（函数名保留 logistic 以兼容旧代码）
    # ==========================================================================

    def solve_logistic_regression(self, X_labeled, Y_labeled, tolerance=None, max_iter=None,
                                  initial_value=None, beta_star=None, CP_if=None,
                                  lambda_range=None, numFolds=None):
        """
        对所有模拟轮次批量求解监督 M-估计参数，并计算评估指标。

        内部调用 solve_logistic_regression_single 完成每轮参数估计，
        然后汇总计算 Bias、SE、MSE 等评估指标。
        函数名保留 logistic 是为了兼容旧主程序；实际损失函数由
        self.model_spec.loss_and_grad 决定。

        参数
        ----
        X_labeled     : np.ndarray，形状 [n, p, T]
            所有模拟轮次的特征数据。
        Y_labeled     : np.ndarray，形状 [n, 1, T]
            所有模拟轮次的响应变量。
        tolerance     : float，可选
            优化收敛容差（梯度范数），默认 5e-3。
        max_iter      : int，可选
            最大迭代次数，默认 500。
        initial_value : np.ndarray，可选
            优化初始值 [p+1, 1]，默认全零向量。
        beta_star     : np.ndarray，可选
            用于计算 Bias/MSE 的参考真值 [p+1, 1]，默认全一向量（仅作占位）。
        CP_if         : int，可选
            是否做交叉验证选 lambda（0=不做，使用 lambda=0），默认 0。
        lambda_range  : np.ndarray，可选
            交叉验证的正则化系数候选范围，默认 logspace(-10, 2, 100)。
        numFolds      : int，可选
            交叉验证折数，默认 5。

        返回
        ----
        beta_hat        : np.ndarray，形状 [p+1, T]
            所有模拟轮次的参数估计值（每列对应一次模拟）。
        Evaluate        : pd.DataFrame
            评估指标表，含 Bias、BIAS_MEAN、SE、SE_MEAN、MSE、MSE_MEAN 列。
        best_lambda_hat : np.ndarray，形状 [T,]
            每次模拟选择的最优正则化系数（CP_if=0 时全为 0）。
        """
        # ---- 参数默认值 ----
        tolerance = 5e-3 if tolerance is None else tolerance
        max_iter = 500 if max_iter is None else max_iter
        initial_value = np.zeros((X_labeled.shape[1] + 1, 1)) if initial_value is None else initial_value
        # beta_star_ref 仅用于计算 Bias/MSE，默认设为全 1 向量（实际使用时应传入真实参数）
        beta_star_ref = (np.vstack([np.ones((1, 1)), np.ones((X_labeled.shape[1], 1))])
                         if beta_star is None else beta_star)
        CP_if = 0 if CP_if is None else CP_if
        lambda_range = np.logspace(-10, 2, 100) if lambda_range is None else lambda_range
        numFolds = 5 if numFolds is None else numFolds

        beta_hat = np.zeros((X_labeled.shape[1] + 1, X_labeled.shape[2]))
        best_lambda_hat = np.zeros(X_labeled.shape[2])

        # 逐轮（每次模拟）单独求解参数
        for t in range(X_labeled.shape[2]):
            X = X_labeled[:, :, t]
            Y = Y_labeled[:, :, t]
            best_lambda, beta_hat_ones = self.solve_logistic_regression_single(
                X, Y, initial_value, {'maxiter': max_iter, 'gtol': tolerance},
                lambda_range, numFolds
            )
            beta_hat[:, t] = beta_hat_ones.ravel()
            best_lambda_hat[t] = best_lambda

        # ---- 计算评估指标 ----
        # Bias：所有轮次估计值与真实值之差的均值（列方向）
        Bias = (1 / X_labeled.shape[2]) * np.sum(beta_hat - beta_star_ref, axis=1).reshape(-1, 1)
        # SE（标准误）：估计值在轮次间的标准差（衡量方差）
        SE   = np.sqrt(np.mean((beta_hat - np.mean(beta_hat, axis=1).reshape(-1, 1)) ** 2, axis=1)).reshape(-1, 1)
        # MSE：均方误差 = 方差 + 偏差²（综合评估）
        MSE  = np.mean((beta_hat - beta_star_ref) ** 2, axis=1).reshape(-1, 1)

        import pandas as pd
        # 将各分量指标扩展为同维度数组，方便 DataFrame 展示
        MSE_MEAN  = np.mean(MSE) * np.ones_like(MSE)
        BIAS_MEAN = np.mean(np.abs(Bias)) * np.ones_like(Bias)
        SE_MEAN   = np.mean(SE) * np.ones_like(SE)
        Evaluate = pd.DataFrame({
            'Bias': Bias.ravel(), 'BIAS_MEAN': BIAS_MEAN.ravel(),
            'SE': SE.ravel(), 'SE_MEAN': SE_MEAN.ravel(),
            'MSE': MSE.ravel(), 'MSE_MEAN': MSE_MEAN.ravel()
        })
        return beta_hat, Evaluate, best_lambda_hat

    def solve_logistic_regression_single(self, X, Y, initial_value, options, lambda_range, numFolds):
        """
        单次参数估计求解（已泛化：使用 self.model_spec.loss_and_grad，正则系数 λ=0）。

        使用 scipy.optimize.minimize（BFGS 方法）最小化 model_spec 定义的损失函数。
        保留原名称以兼容外部调用接口。

        参数
        ----
        X             : np.ndarray，形状 [n, p]
            单次模拟的特征矩阵。
        Y             : np.ndarray，形状 [n, 1] 或 [n,]
            单次模拟的响应变量。
        initial_value : np.ndarray，形状 [p+1, 1]
            优化起始点，默认全零向量。
        options       : dict
            传递给 scipy.optimize.minimize 的优化选项（如 maxiter、gtol）。
        lambda_range  : np.ndarray
            正则化系数候选范围（当前实现未使用交叉验证，保留接口以兼容）。
        numFolds      : int
            交叉验证折数（当前实现未使用，保留接口以兼容）。

        返回
        ----
        best_lambda : float
            最优正则化系数（当前固定为 0.0，不使用正则化）。
        beta_hat    : np.ndarray，形状 [p+1, 1]
            优化得到的参数估计值。
        """
        if initial_value is None:
            initial_value = np.zeros((X.shape[1] + 1, 1))
        if options is None:
            options = {'maxiter': 500, 'gtol': 5e-3}
        best_lambda = 0.0   # 当前不使用正则化（lambda=0），保留接口以备将来扩展
        # 使用 BFGS 拟牛顿方法最小化损失函数，同时利用解析梯度加速收敛
        res = minimize(
            fun=lambda b: self.model_spec.loss_and_grad(b.reshape(-1, 1), X, Y, best_lambda)[0],
            x0=initial_value.ravel(),
            jac=lambda b: self.model_spec.loss_and_grad(b.reshape(-1, 1), X, Y, best_lambda)[1].ravel(),
            method='BFGS', options=options
        )
        return best_lambda, res.x.reshape(-1, 1)

    def objective_function_logistic(self, beta, X_labeled, Y_labeled, lambda_reg):
        """
        目标函数接口（已泛化：委托给 self.model_spec.loss_and_grad）。

        保留原名称以兼容外部调用。内部直接转发给 model_spec，
        不再硬编码逻辑回归负对数似然公式。

        参数
        ----
        beta        : np.ndarray，形状 [p+1, 1]
            当前参数值。
        X_labeled   : np.ndarray，形状 [n, p]
            特征矩阵（不含截距）。
        Y_labeled   : np.ndarray，形状 [n, 1]
            响应变量。
        lambda_reg  : float
            L2 正则化系数。

        返回
        ----
        (f, g) : tuple
            f (float)：目标函数值；g (np.ndarray, [p+1,])：梯度向量。
        """
        return self.model_spec.loss_and_grad(beta, X_labeled, Y_labeled, lambda_reg)

    # ==========================================================================
    # 5. 辅助：可视化
    # ==========================================================================

    def _plot_labeled(self, X, Y):
        """
        绘制有标签数据的前两个特征维度的散点图（用于可视化检验 DGP 是否合理）。

        参数
        ----
        X : np.ndarray，形状 [n, p]
            特征矩阵（取前两列绘图）。
        Y : np.ndarray，形状 [n,]
            响应变量（0/1 二值，决定颜色区分）。
        """
        plt.figure(figsize=(7, 5))
        plt.scatter(X[Y == 1, 0], X[Y == 1, 1], s=30, c='r', alpha=0.6, label='Y=1')
        plt.scatter(X[Y == 0, 0], X[Y == 0, 1], s=30, c='b', alpha=0.6, label='Y=0')
        plt.xlabel('X1'); plt.ylabel('X2')
        plt.title('有标签数据可视化（最后一次模拟）')
        plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    def verify_moments(self, X_unlabeled, mu_T, V_T, tol=0.05, verbose=True):
        """
        验证 source 的样本均值和协方差是否接近目标 (μ_T, V_T)。
        注意正文 Exm4 中 F5–F6 是低阶偏移源，预期不会接近目标矩。

        通过拼合所有模拟轮次的样本来计算大样本均值和协方差，与理论值比较，
        从而验证 _compute_total_moments 和 data_generation_unlabeled_single_source 的正确性。

        参数
        ----
        X_unlabeled : dict
            无标签数据字典，格式为 {source_name: list of [N, p] arrays}。
        mu_T        : np.ndarray，形状 [p, 1]
            目标总体均值（通常为零向量）。
        V_T         : np.ndarray，形状 [p, p]
            目标总体协方差矩阵。
        tol         : float，可选
            可接受的误差阈值（均值用绝对范数，协方差用相对 Frobenius 范数），默认 0.05。
        verbose     : bool，可选
            True = 打印每个 source 的验证结果（含通过/失败符号）。

        返回
        ----
        report : dict
            验证报告字典，格式为：
            {source_name: {'mu_err': float, 'cov_err': float, 'ok': bool}}
            其中 mu_err 为均值绝对误差，cov_err 为协方差相对 Frobenius 误差，
            ok=True 表示两者均在 tol 以内。
        """
        report = {}
        for key, X_list in X_unlabeled.items():
            X_all = np.vstack(X_list)                      # 拼合所有轮次样本，增大样本量提升精度
            mu_hat  = np.mean(X_all, axis=0).reshape(-1, 1)
            cov_hat = np.cov(X_all.T)
            # 均值误差用绝对范数（μ_T 接近 0，不适合做相对误差，避免除以极小值）
            mu_err  = float(np.linalg.norm(mu_hat - mu_T))
            # 协方差误差用相对 Frobenius 范数（加 1e-12 防止除零）
            cov_err = float(np.linalg.norm(cov_hat - V_T, 'fro') / (np.linalg.norm(V_T, 'fro') + 1e-12))
            ok = (mu_err < tol) and (cov_err < tol)        # 两个误差均需在阈值内
            report[key] = {'mu_err': mu_err, 'cov_err': cov_err, 'ok': ok}
            if verbose:
                status = '✓' if ok else '✗'
                print(f"  [{status}] {key:8s}  |μ̂-μ_T|={mu_err:.4f}  "
                      f"|Σ̂-V_T|_F/|V_T|_F={cov_err:.4f}")
        return report
