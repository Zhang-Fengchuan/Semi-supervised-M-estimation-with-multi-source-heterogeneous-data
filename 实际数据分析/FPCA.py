import numpy as np
import matplotlib.pyplot as plt


def _trapezoid_weights(t):
    """
    计算梯形积分权重向量。

    对给定的时间网格 t，使用复合梯形公式为每个时间点分配积分权重，
    使得 sum(w_i * f(t_i)) ≈ ∫f(t)dt。
    端点权重为相邻区间长度的一半，内部点权重为左右区间长度之和的一半。

    参数
    ----
    t : array-like, shape (T,)
        严格单调递增的时间网格点序列。

    返回
    ----
    w : np.ndarray, shape (T,)
        每个时间点对应的梯形积分权重，满足 sum(w) ≈ t[-1] - t[0]。

    异常
    ----
    ValueError
        若时间网格点数 < 2，或权重存在非正值（说明 t 非严格递增）。
    """
    t = np.asarray(t, dtype=float)
    T = t.size
    if T < 2:
        raise ValueError("time grid must have >=2 points")
    w = np.zeros(T, dtype=float)
    # 左端点：只有右侧半个区间贡献权重
    w[0]  = (t[1]-t[0])/2.0
    # 右端点：只有左侧半个区间贡献权重
    w[-1] = (t[-1]-t[-2])/2.0
    if T > 2:
        # 内部点：左右各半个区间，等价于 (t[i+1] - t[i-1]) / 2
        w[1:-1] = (t[2:] - t[:-2]) / 2.0
    if not np.all(w > 0):
        raise ValueError("weights non-positive: t must be strictly increasing")
    return w


class FPCA1D:
    """
    一维函数型主成分分析（Functional PCA, FPCA）。

    针对角度时间序列数据 θ(t) 进行函数型 PCA，将每条曲线投影到
    低维得分空间，同时保留曲线在 L²(w) 内积意义下的最大方差方向。

    输入数据形状：(n, T) 或 (n, T, 1)，其中 n 为样本数，T 为时间点数。

    参数
    ----
    n_components : int 或 None
        保留的主成分数量。若为 None，则由 pve 自动确定。
    pve : float, 默认 0.95
        目标累计方差贡献率（proportion of variance explained），
        仅当 n_components=None 时生效，取值范围 (0, 1]。
    center : bool, 默认 True
        是否在 PCA 前对曲线去均值（减去样本均值函数）。
    unbiased_var : bool, 默认 False
        计算特征值时是否使用无偏估计（除以 n-1 而非 n）。

    属性（fit 后可用）
    -----------------
    t_ : np.ndarray, shape (T,)
        拟合时使用的时间网格。
    w_ : np.ndarray, shape (T,)
        梯形积分权重向量。
    mean_ : np.ndarray, shape (T,)
        样本均值函数（center=True 时为实际均值，否则为零向量）。
    phi_ : np.ndarray, shape (T, K)
        前 K 个主成分特征函数，在 L²(w) 范数下已归一化。
    scores_ : np.ndarray, shape (n, K)
        训练数据在前 K 个特征函数上的投影得分。
    eigenvalues_ : np.ndarray, shape (K,)
        前 K 个特征值（各主成分方向上的方差）。
    explained_ratio_ : np.ndarray, shape (K,)
        前 K 个主成分各自的方差贡献率。
    """

    def __init__(self, n_components=None, pve=0.95, center=True, unbiased_var=False):
        """
        初始化 FPCA 模型配置；真正的主成分在 fit() 中计算。

        参数
        ----
        n_components : int 或 None
            固定保留的主成分数量。若为 None，则根据 pve 自动选择。
        pve : float, 默认 0.95
            当 n_components=None 时使用的累计方差贡献率阈值。
        center : bool, 默认 True
            是否先减去样本均值函数。
        unbiased_var : bool, 默认 False
            True 时特征值除以 n-1；False 时除以 n。
        """
        self.n_components  = n_components   # 主成分数量（None 表示自动按 pve 确定）
        self.pve           = pve            # 目标累计方差贡献率
        self.center        = center         # 是否去均值
        self.unbiased_var  = unbiased_var   # 是否使用无偏方差估计
        # 以下属性在 fit() 后被赋值
        self.t_ = None
        self.w_ = None
        self.mean_ = None          # (T,)  样本均值函数
        self.phi_  = None          # (T, K) 主成分特征函数矩阵
        self.scores_ = None        # (n, K) 主成分得分矩阵
        self.eigenvalues_ = None   # (K,)   特征值向量
        self.explained_ratio_ = None  # (K,) 方差贡献率向量

    def _coerce_theta(self, data):
        """
        将输入数据强制转换为二维数组 (n, T)。

        支持两种输入格式：
        - (n, T)：直接返回
        - (n, T, 1)：压缩最后一维后返回

        参数
        ----
        data : array-like
            输入的角度时间序列数据。

        返回
        ----
        arr : np.ndarray, shape (n, T)
            标准化后的二维数组。

        异常
        ----
        ValueError
            若输入形状不符合 (n, T) 或 (n, T, 1)。
        """
        arr = np.asarray(data, dtype=float)
        if arr.ndim == 2:
            return arr                      # (n, T) 直接返回
        if arr.ndim == 3 and arr.shape[2] == 1:
            return arr[..., 0]              # (n, T, 1) -> 压缩最后维度 -> (n, T)
        raise ValueError("theta must be (n, T) or (n, T, 1)")

    def fit(self, data, t=None):
        """
        在给定数据上拟合函数型 PCA 模型。

        算法流程：
        1. 对曲线矩阵去均值（可选）
        2. 计算梯形积分权重 w，构造加权变换矩阵 W^{1/2}
        3. 对加权矩阵 G = Θ^c · W^{1/2} 做 SVD
        4. 根据累计方差贡献率或指定数量选择主成分数 K
        5. 将右奇异向量变换回原时域，并按 L²(w) 范数归一化得到特征函数
        6. 计算每个样本在各特征函数上的投影得分

        参数
        ----
        data : array-like, shape (n, T) 或 (n, T, 1)
            训练用角度时间序列数据，n 为样本数，T 为时间点数。
        t : array-like, shape (T,), 可选
            时间网格点，默认为 [0, 1] 上均匀分布的 T 个点。

        返回
        ----
        self : FPCA1D
            拟合后的模型对象（支持链式调用）。
        """
        Theta = self._coerce_theta(data)    # 统一转为 (n, T)
        n, T = Theta.shape
        if t is None:
            # 若未提供时间网格，默认使用 [0,1] 上的均匀网格
            t = np.linspace(0.0, 1.0, T)
        t = np.asarray(t, dtype=float)

        # 步骤 1：去均值（减去样本均值函数），使协方差矩阵以零为中心
        self.mean_ = Theta.mean(axis=0) if self.center else np.zeros(T)
        Tc = Theta - self.mean_             # 去均值后的曲线矩阵 (n, T)

        # 步骤 2：计算梯形权重 w 并构造 W^{1/2} 和 W^{-1/2}
        # 权重向量 w 使得离散内积 <f,g> = sum(w_i * f_i * g_i) 逼近连续 L² 内积
        w  = _trapezoid_weights(t)          # (T,)
        sw = np.sqrt(w)                     # W^{1/2}，用于构造加权矩阵
        isw = 1.0 / sw                      # W^{-1/2}，用于后续还原特征函数

        # 步骤 3：构造加权数据矩阵 G = Θ^c · W^{1/2}，并对其做 SVD
        # SVD(G) = U·S·Vt，其中 G 的右奇异向量 V 对应加权空间中的主方向
        G = Tc * sw[None, :]                # 按列乘以 W^{1/2}，得到 (n, T)
        U, S, Vt = np.linalg.svd(G, full_matrices=False)
        V = Vt.T                            # 右奇异向量矩阵 (T, T)

        # 步骤 4：确定保留的主成分数 K
        # 特征值 = 奇异值² / (n 或 n-1)，表示各方向上的方差
        denom = (n-1) if (self.unbiased_var and n > 1) else n
        eig_all = (S**2) / denom            # 所有主成分对应的特征值
        cum = np.cumsum(eig_all) / np.sum(eig_all)  # 累计方差贡献率
        if self.n_components is None:
            # 自动选取使累计方差贡献率达到 pve 所需的最小主成分数
            K = int(np.searchsorted(cum, float(self.pve)) + 1)
        else:
            K = int(self.n_components)
        K = max(1, min(K, V.shape[1]))      # 保证 K 在合法范围内

        # 步骤 5：将加权空间中的右奇异向量还原为原始时域的特征函数
        # phi = W^{-1/2} · V，再按 L²(w) 范数归一化
        Vphi = V[:, :K] * isw[:, None]      # W^{-1/2} * V，shape (T, K)
        # 计算每个特征函数在 L²(w) 内积下的范数
        norms = np.sqrt((w[:, None] * (Vphi**2)).sum(axis=0))
        Vphi /= norms[None, :]              # 归一化，使 <phi_k, phi_k>_w = 1

        # 步骤 6：计算主成分得分（等价于 U * S 的前 K 列）
        # 得分 = <Theta_i - mean, phi_k>_w，表示第 i 个样本沿第 k 个特征函数的投影幅度
        scores = U[:, :K] * S[:K][None, :]

        # 存储拟合结果
        self.t_ = t
        self.w_ = w
        self.phi_ = Vphi                     # 主成分特征函数 (T, K)
        self.scores_ = scores                # 主成分得分 (n, K)
        self.eigenvalues_ = eig_all[:K]      # 前 K 个特征值
        self.explained_ratio_ = self.eigenvalues_ / eig_all.sum()  # 方差贡献率
        return self

    def transform(self, data):
        """
        将新数据投影到已拟合的主成分空间，得到低维得分向量。

        对每条新曲线先减去训练均值，再与各特征函数做 L²(w) 内积，
        得到该曲线在各主成分方向上的投影系数（得分）。

        参数
        ----
        data : array-like, shape (n_new, T) 或 (n_new, T, 1)
            待投影的新样本角度时间序列，时间点数须与训练数据一致。

        返回
        ----
        scores : np.ndarray, shape (n_new, K)
            每个新样本在前 K 个主成分方向上的投影得分。

        异常
        ----
        RuntimeError
            若模型尚未拟合（未调用 fit()）。
        ValueError
            若新数据的时间点数与训练数据不一致。
        """
        if self.phi_ is None:
            raise RuntimeError("fit() first")
        Theta = self._coerce_theta(data)    # (n_new, T)
        if Theta.shape[1] != self.t_.size:
            raise ValueError("shape mismatch with fitted model")
        # 去均值：减去训练集计算的均值函数
        Tc = Theta - self.mean_[None, :]
        # 计算 L²(w) 内积：<Tc_i, phi_k>_w = sum_t w_t * Tc_i(t) * phi_k(t)
        part = Tc[:, :, None] * self.phi_[None, :, :]   # 广播乘法 (n, T, K)
        scores = (part * self.w_[None, :, None]).sum(axis=1)  # 对时间维求和 (n, K)
        return scores                                   # (n, K)

    def reconstruct(self, scores=None, components=None):
        """
        从主成分得分重建（近似）原始曲线。

        使用前 K_use 个主成分线性组合来重建曲线：
        θ̂_i(t) = mean(t) + sum_{k=1}^{K_use} score_{i,k} * phi_k(t)

        参数
        ----
        scores : np.ndarray, shape (n, K) 或 None
            用于重建的主成分得分。若为 None，则使用训练时计算的 scores_。
        components : int 或 None
            重建时使用的主成分数量，须 <= K（拟合时保留的总数）。
            若为 None，则使用全部 K 个主成分。

        返回
        ----
        recon : np.ndarray, shape (n, T)
            重建后的曲线矩阵（已加回均值函数）。

        异常
        ----
        RuntimeError
            若模型尚未拟合（未调用 fit()）。
        """
        if self.phi_ is None:
            raise RuntimeError("fit() first")
        # 若未提供得分，使用训练集得分
        Xi = self.scores_ if scores is None else np.asarray(scores, dtype=float)
        K_total = self.phi_.shape[1]
        # 确定实际使用的主成分数，限制在 [1, K_total] 范围内
        K_use = K_total if components is None else int(min(max(1, components), K_total))
        # 线性重建：score (n, K_use) × phi^T (K_use, T) -> (n, T)
        recon = np.tensordot(Xi[:, :K_use], self.phi_[:, :K_use].T, axes=(1, 0))  # (n, T)
        # 加回均值函数，还原到原始空间
        recon += self.mean_[None, :]
        return recon


# if __name__ == "__main__":
#     rng = np.random.default_rng(0)
#     n, T = 10, 600
#     t = np.linspace(0, 1, T)
#     phi1 = np.sin(2*np.pi*t)        # 一个潜在时间模式
#     z = rng.normal(size=n)
#     theta = 0.2*rng.normal(size=(n,T)) + z[:,None]*phi1[None,:]
#     fp = FPCA1D(n_components=10).fit(theta, t=t) #pve=0.95
#     print("scores:", fp.scores_.shape, "first explained ratio:", fp.explained_ratio_[0])
#     plt.figure()
#     t = np.arange(10)
#     plt.plot(t, fp.scores_.T)  # 把 (n, T) 转成 (T, n)
#     plt.xlabel("time")
#     plt.ylabel("theta")
#     plt.title("theta(t) — all samples")
#     plt.show()
