"""
SelectionViz.py — 选择频次可视化模块
======================================
本模块专门负责将半监督数据源筛选结果可视化，供 MstMdsp_simulation_main.py 调用。
完全独立于仿真计算逻辑，仅接受已计算好的选择频次矩阵 select_times_pro 作为输入。

对外公开接口
-----------
plot_selection(select_times_pro, h_mu, h_sigma,
               which_Exm, sample_size_n, sample_size_N, simulation_times)
    为单次（单个 n）的仿真结果生成：
      ① Python 热力图（bicubic 双三次插值平滑，带 z=0 等值线）
      ② Python 3D 条形图（winter 配色，高值=绿，低值=蓝）
      ③ 上述两图拼合的组合图
      ④ 导出 .mat 数据文件 + MATLAB 绘图脚本并自动执行 MATLAB（如已安装）

plot_multi_panel(results_dict, which_Exm, sample_size_N, simulation_times, code_dir)
    将多个样本量 n 的结果拼成论文用 2 行 × N 列 大图（上行热力图，下行3D柱）。
    results_dict 格式：{n: (select_times_pro, h_mu, h_sigma)}

配色约定
--------
使用 matplotlib 的 winter colormap（蓝→绿）：
    低频次（被选中次数少） → 蓝色
    高频次（被选中次数多） → 绿色
与论文 MATLAB 版本保持一致（均用 winter，不翻转）。

坐标轴排列约定
--------------
ε 轴（均值扰动 / 方差扰动）按"中心对称排列"：
    |ε| 最小（最接近0，即最干净）的数据源放在轴的中间位置，
    |ε| 最大（偏差最大）的数据源放在两端。
这样视觉上能直观看到"中间高、两端低"的频次分布模式。
"""

import os
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use('Agg')          # 无头模式（服务器环境无显示器）
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import RectBivariateSpline
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — 注册 3D 投影，必须导入

# ── 全局字体：Times New Roman ──────────────────────────────────────────────────
# 论文图表要求使用 Times New Roman 衬线字体；
# mathtext.fontset='stix' 保证数学公式（如 ε^μ）也使用衬线风格。
plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset':   'stix',
    'axes.unicode_minus': False,   # 防止负号显示为方块
})

# 当前整理版根目录，作为默认输出路径。
_CODE_DIR = str(Path(__file__).resolve().parents[1])
_OUTPUTS_DIR = os.path.join(_CODE_DIR, "数值模拟", "模拟结果")   # 所有可视化结果统一输出根目录


def _make_run_dir(which_Exm, sample_size_n, sample_size_N, simulation_times,
                  outputs_root=None):
    """
    为单次仿真实验创建专属输出文件夹，命名编码所有关键超参数。

    目录结构示例：
        <outputs_root>/exm2_n250_N5000_T100/

    Parameters
    ----------
    which_Exm        : int    — 实验编号
    sample_size_n    : int    — 标记样本数 n
    sample_size_N    : int    — 每个无标签源的样本数 N
    simulation_times : int    — 蒙特卡洛重复次数 T
    outputs_root     : str    — 输出根目录，None 时用 _OUTPUTS_DIR

    Returns
    -------
    run_dir : str
        创建好的（如不存在则自动 mkdir）专属输出目录绝对路径。
    """
    if outputs_root is None:
        outputs_root = _OUTPUTS_DIR
    run_dir = os.path.join(
        outputs_root,
        f'exm{which_Exm}_n{sample_size_n}_N{sample_size_N}_T{simulation_times}'
    )
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

# ── colormap（high=绿，low=蓝，与论文 winter 一致）─────────────────────────────
# winter: 0.0（最小）→ 蓝；1.0（最大）→ 绿
# 热力图和3D柱均使用同一 colormap，保证颜色含义统一
_CMAP_HEAT = plt.cm.winter   # 热力图用
_CMAP_BAR  = plt.cm.winter   # 3D 柱状图用


# =============================================================================
#  内部工具函数
# =============================================================================

def _center_sort_idx(vals):
    """
    计算"中心对称排列"的索引，使 |vals| 最小的元素落在序列正中央，
    |vals| 最大的元素落在序列两端。

    用途：将 ε 轴上的扰动参数按"偏差从小到大向两端扩散"的顺序排列，
          视觉上让"最干净"的数据源（ε≈0）位于图的中心。

    Parameters
    ----------
    vals : array-like of float
        原始参数值列表（可含负值），长度为 n。

    Returns
    -------
    result : np.ndarray of int, shape (n,)
        排列后的原始索引数组。result[i] 表示排列后第 i 个位置对应原数组的哪个元素。
        中间位置（n//2 及邻近）的元素 |vals| 最小。

    Example
    -------
    >>> _center_sort_idx([-3, -1, 0, 1, 3])
    array([4, 2, 0, 1, 3])   # 中心是索引1(val=-1)和2(val=0)，两端是0和4
    """
    n      = len(vals)
    # by_abs: 按绝对值从小到大排序后的原始索引（最小 |val| 排在最前）
    by_abs = np.argsort(np.abs(vals))
    result = np.empty(n, dtype=int)
    # lo 从中心向左扩展，hi 从中心向右扩展
    lo, hi = n // 2 - 1, n // 2
    for i, orig_idx in enumerate(by_abs):
        if i % 2 == 0:
            # 偶数轮：放在右侧（hi 位置），然后 hi 右移
            result[hi] = orig_idx
            hi += 1
        else:
            # 奇数轮：放在左侧（lo 位置），然后 lo 左移
            result[lo] = orig_idx
            lo -= 1
    return result


def _fmt_eps(v):
    """
    将扰动参数值格式化为简洁的字符串，用作坐标轴刻度标签。

    规则：
      |v| < 0.1  → 保留两位小数（如 0.05 → '0.05'）
      |v| < 10   → 保留一位小数（如  1.5 → '1.5'）
      |v| ≥ 10   → 取整（如 10.0 → '10'）

    Parameters
    ----------
    v : float
        扰动参数值。

    Returns
    -------
    str
        格式化后的字符串。
    """
    if abs(v) < 0.1:
        return f'{v:.2f}'
    return f'{v:.1f}' if abs(v) < 10 else f'{v:.0f}'


def _prep_data(select_times_pro, h_mu, h_sigma, simulation_times):
    """
    将原始选择频次矩阵转换为所有绘图函数共用的数据包（dict）。

    本函数完成以下预处理步骤：
      1. 按"中心对称"重排 ε^μ 和 ε^Σ 轴，使最干净的数据源居中
      2. 对选择频次矩阵进行归一化（以中值为中心，缩放至 [-1, 1]）
      3. 用双三次样条插值生成 300×300 的平滑网格（供热力图 imshow 使用）
      4. 为 3D 柱状图准备颜色（根据频次高低映射 winter colormap）
      5. 准备 colorbar 的 ScalarMappable 对象和 z 轴刻度

    Parameters
    ----------
    select_times_pro : array-like, shape (n_mu, n_sigma)
        各数据源在仿真中被选中的次数矩阵。
        行对应 ε^μ 的取值，列对应 ε^Σ 的取值。
    h_mu : list of float
        均值扰动参数 ε^μ 的取值列表，长度 = n_mu。
    h_sigma : list of float
        方差扰动参数 ε^Σ 的取值列表，长度 = n_sigma。
    simulation_times : int
        蒙特卡洛重复总次数，用于确定 colorbar 的最大值。

    Returns
    -------
    D : dict，包含以下键值：
        n_mu, n_sigma     : int   — ε^μ 和 ε^Σ 的网格点数
        h_mu_disp         : (n_mu,)   — 重排后的 ε^μ 值
        h_sigma_disp      : (n_sigma,) — 重排后的 ε^Σ 值
        xi_mu, xi_sigma   : (n_mu,), (n_sigma,) — 整数坐标（从1开始）
        mu_labels         : list of str — ε^μ 刻度标签
        sigma_labels      : list of str — ε^Σ 刻度标签
        ST_disp           : (n_mu, n_sigma) — 重排后的原始频次矩阵
        ST_interp         : (300, 300)      — 插值平滑后的归一化矩阵（[-1,1]）
        xpos, ypos        : (n_mu*n_sigma,) — 3D 柱的 x/y 底面中心坐标（展平）
        dz                : (n_mu*n_sigma,) — 3D 柱的高度（选中次数，展平）
        bar_colors        : (n_mu*n_sigma, 4) — 各柱的 RGBA 颜色
        sm                : ScalarMappable  — 用于生成 colorbar
        zticks            : array           — z 轴刻度位置
    """
    _h_mu    = np.array(h_mu,    dtype=float)
    _h_sigma = np.array(h_sigma, dtype=float)
    n_mu     = len(_h_mu)
    n_sigma  = len(_h_sigma)
    ST       = np.array(select_times_pro, dtype=float)  # shape (n_mu, n_sigma)

    # ── 步骤1：中心对称重排 ε 轴 ──────────────────────────────────────────────
    mu_idx    = _center_sort_idx(_h_mu)     # 重排后的行索引
    sigma_idx = _center_sort_idx(_h_sigma)  # 重排后的列索引
    h_mu_d    = _h_mu[mu_idx]              # 重排后的 ε^μ 值序列
    h_sigma_d = _h_sigma[sigma_idx]        # 重排后的 ε^Σ 值序列
    ST_disp   = ST[np.ix_(mu_idx, sigma_idx)]  # 同步重排频次矩阵

    # 坐标轴刻度：整数位置（1, 2, ..., n_mu/n_sigma）
    mu_labels    = [_fmt_eps(v) for v in h_mu_d]
    sigma_labels = [_fmt_eps(v) for v in h_sigma_d]
    xi_mu    = np.arange(1, n_mu    + 1, dtype=float)
    xi_sigma = np.arange(1, n_sigma + 1, dtype=float)

    # ── 步骤2：归一化频次至 [-1, 1] ───────────────────────────────────────────
    # 以矩阵最大值的一半为中心（而非均值），保证0对应"中等频次"
    _half       = ST_disp.max() / 2.0
    ST_c        = ST_disp - _half
    _vmax       = np.abs(ST_c).max() + 1e-12   # 防止除零
    ST_norm     = ST_c / _vmax                  # 归一化到 [-1, 1]

    # ── 步骤3：双三次样条插值平滑 ─────────────────────────────────────────────
    # RectBivariateSpline：在规则网格上的双变量样条，kx=ky=3 为三阶（双三次）
    # 将稀疏的 n_mu×n_sigma 网格插值到 300×300 的精细网格，使热力图更平滑
    _interp   = RectBivariateSpline(xi_mu, xi_sigma, ST_norm, kx=3, ky=3)
    ST_interp = np.clip(
        _interp(np.linspace(1, n_mu, 300), np.linspace(1, n_sigma, 300)),
        -1, 1   # 裁剪到 [-1, 1]，防止插值超出范围
    )

    # ── 步骤4：3D 柱状图数据准备 ──────────────────────────────────────────────
    # 用 meshgrid 生成所有 (μ方向, σ方向) 网格点的坐标，再展平成一维数组
    MU_g, SIG_g = np.meshgrid(xi_mu, xi_sigma, indexing='ij')  # (n_mu, n_sigma)
    xpos = MU_g.ravel()          # 柱的 x 中心坐标（展平）
    ypos = SIG_g.ravel()         # 柱的 y 中心坐标（展平）
    dz   = ST_disp.ravel().astype(float)  # 柱的高度 = 选中次数（展平）

    # 颜色映射：将频次 [0, simulation_times] 线性映射到 winter colormap
    _norm_dz   = plt.Normalize(vmin=0, vmax=max(simulation_times, dz.max()))
    bar_colors = _CMAP_BAR(_norm_dz(dz))  # RGBA 颜色数组

    # ── 步骤5：colorbar 辅助对象 ──────────────────────────────────────────────
    sm = cm.ScalarMappable(cmap=_CMAP_BAR, norm=_norm_dz)
    sm.set_array([])  # matplotlib 要求显式调用，否则 colorbar 报错

    # z 轴刻度：从0到 simulation_times，均匀分布约8个刻度
    zticks = np.arange(0, simulation_times + 1, max(10, simulation_times // 8))

    return dict(
        n_mu=n_mu, n_sigma=n_sigma,
        h_mu_disp=h_mu_d, h_sigma_disp=h_sigma_d,
        xi_mu=xi_mu, xi_sigma=xi_sigma,
        mu_labels=mu_labels, sigma_labels=sigma_labels,
        ST_disp=ST_disp, ST_interp=ST_interp,
        xpos=xpos, ypos=ypos,
        dz=dz, bar_colors=bar_colors, sm=sm, zticks=zticks,
    )


# =============================================================================
#  绘图函数（接受数据包，在调用方已创建的 axes 上绘制）
# =============================================================================

def _draw_heatmap(ax, fig_ref, D, title_str, panel_label=None):
    """
    在给定的 2D Axes 上绘制选择频次热力图。

    特性：
      · 使用双三次插值后的平滑矩阵 D['ST_interp']，避免方格感
      · winter colormap（蓝=低频次，绿=高频次）
      · 在 z=0 处叠加黑色等值线（LineWidth=3），直观区分"倾向选中"和"倾向不选"的区域
      · colorbar 刻度固定为 [-1, -0.5, 0, 0.5, 1]（归一化尺度）

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        2D 坐标系对象，由调用方提供。
    fig_ref : matplotlib.figure.Figure
        当前图形对象，用于添加 colorbar（必须与 ax 所在 figure 一致）。
    D : dict
        由 _prep_data() 返回的数据包。
    title_str : str
        子图标题，支持 LaTeX 格式（如 '$n_0=250$'）。
    panel_label : str or None
        面板编号标签（如 '(a)'），显示在左上角；None 表示不添加。

    Returns
    -------
    im : AxesImage
        imshow 返回的图像对象（可用于后续调整）。
    """
    xi_mu, xi_sigma = D['xi_mu'], D['xi_sigma']

    # ── 主体：双三次插值后的平滑热力图 ──────────────────────────────────────
    # ST_interp 的形状为 (300, 300)，.T 转置后行=σ轴，列=μ轴，对应 imshow 的 (y, x)
    # origin='lower' 使 y 轴从下往上增大（符合数学习惯）
    # extent 将像素坐标映射到 xi_mu / xi_sigma 的实际刻度范围
    im = ax.imshow(
        D['ST_interp'].T, origin='lower', aspect='auto',
        cmap=_CMAP_HEAT, vmin=-1, vmax=1,
        extent=[xi_mu.min()-0.5, xi_mu.max()+0.5,
                xi_sigma.min()-0.5, xi_sigma.max()+0.5]
    )

    # ── 叠加 z=0 等值线（黑色粗线，区分正负区域）────────────────────────────
    # levels=[0.0] 表示只画 ST_interp=0 的等高线
    # colors='black', linewidths=3.0 与论文 MATLAB 版本保持一致
    ax.contour(
        np.linspace(xi_mu.min()-0.5, xi_mu.max()+0.5, 300),
        np.linspace(xi_sigma.min()-0.5, xi_sigma.max()+0.5, 300),
        D['ST_interp'].T, levels=[0.0], colors='black', linewidths=3.0
    )

    # ── 坐标轴标签和刻度 ─────────────────────────────────────────────────────
    # 刻度直接用整数索引 1..n（与 MATLAB 版本一致；ε 数值差异微小且常为负，渲染易拥挤）
    ax.set_xticks(xi_mu);    ax.set_xticklabels([str(int(v)) for v in xi_mu],    fontsize=10)
    ax.set_yticks(xi_sigma); ax.set_yticklabels([str(int(v)) for v in xi_sigma], fontsize=10)
    ax.set_xlabel(r'$\epsilon^{\mu}$',    fontsize=13)
    ax.set_ylabel(r'$\epsilon^{\Sigma}$', fontsize=13)
    ax.set_title(title_str, fontsize=11, pad=6)
    ax.set_xlim(xi_mu.min()-0.5, xi_mu.max()+0.5)
    ax.set_ylim(xi_sigma.min()-0.5, xi_sigma.max()+0.5)

    # ── colorbar ─────────────────────────────────────────────────────────────
    cbar = fig_ref.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.ax.tick_params(labelsize=8)

    # ── 面板标签（(a), (b), ...）─────────────────────────────────────────────
    if panel_label:
        ax.text(-0.14, 1.04, panel_label, transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top')
    return im


def _draw_3d_bars(ax3, D, title_str, simulation_times, panel_label=None):
    """
    在给定的 3D Axes 上绘制选择频次 3D 柱状图（bar3d）。

    特性：
      · winter colormap（蓝=低频次，绿=高频次），与热力图一致
      · 每根柱子顶端标注具体数值（fontsize=7），仅对 dz>0 的柱子标注
      · 视角固定为 elev=30°, azim=-50°（与论文视角一致）
      · 背景面板设为透明，网格线设为淡灰色虚线

    Parameters
    ----------
    ax3 : mpl_toolkits.mplot3d.axes3d.Axes3D
        3D 坐标系对象，由调用方提供。
    D : dict
        由 _prep_data() 返回的数据包。
    title_str : str
        子图标题，支持 LaTeX 格式。
    simulation_times : int
        蒙特卡洛重复总次数，用于设置 z 轴上限（给柱顶标注留出空间）。
    panel_label : str or None
        面板编号标签（如 '(e)'），显示在左上角；None 表示不添加。
    """
    dx = dy = 0.65   # 柱子的宽度和深度（占格子的 65%，留出间隙便于区分）

    # ── 绘制所有柱子 ─────────────────────────────────────────────────────────
    # bar3d 参数说明：
    #   (x, y, z) = 柱底面左下角坐标（x-dx/2 实现居中对齐）
    #   dx, dy, dz = 柱的宽度、深度、高度
    #   color = 各柱的颜色（由频次映射 winter 得到）
    #   shade=True 开启光影效果；edgecolor='white' 柱边缘白色便于区分
    ax3.bar3d(D['xpos']-dx/2, D['ypos']-dy/2, np.zeros(len(D['xpos'])),
              dx, dy, D['dz'], color=D['bar_colors'], shade=True,
              edgecolor='white', linewidth=0.4, zsort='average')

    # ── 柱顶数值标注 ─────────────────────────────────────────────────────────
    # 只标注 dz>0 的柱子（频次为0则不标注，保持图面整洁）
    # 偏移量 +1.5 使标注文字与柱顶有一定间距，不与柱体重叠
    for xi_, yi_, dzi_ in zip(D['xpos'], D['ypos'], D['dz']):
        if dzi_ > 0:
            ax3.text(xi_, yi_, float(dzi_) + 1.5, str(int(dzi_)),
                     ha='center', va='bottom', fontsize=7, color='black')

    # ── 坐标轴刻度和标签 ─────────────────────────────────────────────────────
    # 与热力图一致：刻度用整数索引 1..n（不用 ε 数值）
    xi_mu, xi_sigma = D['xi_mu'], D['xi_sigma']
    ax3.set_xticks(xi_mu);    ax3.set_xticklabels([str(int(v)) for v in xi_mu],    fontsize=9)
    ax3.set_yticks(xi_sigma); ax3.set_yticklabels([str(int(v)) for v in xi_sigma], fontsize=9)
    ax3.set_xlabel(r'$\epsilon^{\mu}$',    fontsize=11, labelpad=8)
    ax3.set_ylabel(r'$\epsilon^{\Sigma}$', fontsize=11, labelpad=8)
    ax3.set_zlabel('Select Count',          fontsize=10, labelpad=6)

    # z 轴范围：上限为 simulation_times+2，为柱顶标注留出空间
    ax3.set_zlim(0, simulation_times + 2)
    ax3.set_zticks(D['zticks'])
    ax3.set_zticklabels([str(int(z)) for z in D['zticks']], fontsize=8)
    ax3.set_title(title_str, fontsize=11, pad=4)

    # ── 视角和背景样式 ────────────────────────────────────────────────────────
    ax3.view_init(elev=30, azim=-50)  # 固定视角，论文标准视角
    for pane in (ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane):
        pane.fill = False                     # 背景面板透明
        pane.set_edgecolor('lightgray')       # 面板边缘浅灰色
    ax3.grid(True, linestyle='--', alpha=0.3)

    # ── 面板标签 ─────────────────────────────────────────────────────────────
    if panel_label:
        ax3.text2D(-0.08, 1.03, panel_label, transform=ax3.transAxes,
                   fontsize=14, fontweight='bold', va='top')


# =============================================================================
#  MATLAB 高质量版本导出
# =============================================================================

def _export_matlab(D, which_Exm, sample_size_n, sample_size_N,
                   simulation_times, code_dir):
    """
    将绘图数据导出为 .mat 文件，并生成 MATLAB 绘图脚本，最后自动调用 MATLAB 执行。

    生成的文件：
      · select_data_exm{E}_n{n}.mat      — scipy.io.savemat 导出的数据文件
      · plot_select_exm{E}_n{n}.m        — MATLAB 绘图脚本（含热力图+3D柱+组合图）
      · select_heatmap_matlab_exm{E}_n{n}.png  — MATLAB 绘制的热力图（300 dpi）
      · select_3d_matlab_exm{E}_n{n}.png       — MATLAB 绘制的 3D 柱状图（300 dpi）
      · select_combo_matlab_exm{E}_n{n}.png    — MATLAB 绘制的组合图（300 dpi）

    MATLAB 脚本特性：
      · set(0,'DefaultAxesFontName','Times New Roman') — 全局 Times New Roman
      · colormap(gca, winter) — 与 Python 版配色一致（必须指定 axes 句柄，否则影响错图）
      · z=0 等值线用 LineWidth=3 黑色实线
      · 3D 柱颜色由 ZData 映射（b.CData = zdata; b.FaceColor = 'interp'）
      · 柱顶数值标注：FontSize=8，仅 ST(i,j)>0 时标注，偏移 +1.5

    Parameters
    ----------
    D : dict
        由 _prep_data() 返回的数据包。
    which_Exm : int
        实验编号，用于文件命名。
    sample_size_n : int
        标记样本数，用于文件命名和图标题。
    sample_size_N : int
        无标签样本数，用于图标题。
    simulation_times : int
        蒙特卡洛重复次数，用于 z 轴范围和 colorbar 上限。
    code_dir : str
        输出文件的目标目录（.mat 文件和 .m 脚本均保存至此）。
    """
    import scipy.io as sio

    # ── 步骤1：保存 .mat 数据文件 ─────────────────────────────────────────────
    # savemat 要求所有数值型数据为 numpy array；
    # h_mu_disp/h_sigma_disp 需 reshape 为列向量 (-1,1) 以符合 MATLAB 列向量惯例
    mat_data = {
        'ST_disp':          D['ST_disp'],
        'h_mu_disp':        D['h_mu_disp'].reshape(-1, 1),
        'h_sigma_disp':     D['h_sigma_disp'].reshape(-1, 1),
        'n_mu':             float(D['n_mu']),
        'n_sigma':          float(D['n_sigma']),
        'simulation_times': float(simulation_times),
        'sample_size_n':    float(sample_size_n),
        'sample_size_N':    float(sample_size_N),
        'which_Exm':        float(which_Exm),
    }
    # code_dir 在新版结构下传入的是「单次实验专属目录」（由 _make_run_dir 创建），
    # 因此文件名不再需要 exm/n 后缀，目录本身已编码超参数。
    mat_path = os.path.join(code_dir, 'data.mat')
    sio.savemat(mat_path, mat_data)
    print(f'[MATLAB] 数据已保存: {mat_path}')

    matlab_out_dir = os.path.abspath(code_dir).replace("'", "''")

    # ── 步骤2：生成 MATLAB 绘图脚本字符串 ────────────────────────────────────
    # 关键设计决策（针对 v2 用户反馈调整）：
    #   · 标题用 'Interpreter', 'tex'（不是 latex）→ 避免 $...$ 解析失败,
    #     `n_0`/`n_k` 在 tex 解释器下自然渲染为下标，更稳定
    #   · 坐标轴刻度直接用 1..n 整数索引，不再用 ε 数值作为刻度文本
    #     （ε 值差异微小且常含负数，渲染容易拥挤）
    #   · 每张图额外 savefig 出 .fig 文件，方便在 MATLAB 中后续手动调整
    m_script = f"""% =========================================================
%  MATLAB 绘图脚本 — 选择频次热力图 + 3D 直方图
%  由 SelectionViz.py 自动生成
% =========================================================
clear; close all; clc;

%% 1. 全局字体（设置后所有新建图形均使用 Times New Roman）
set(0, 'DefaultAxesFontName', 'Times New Roman');
set(0, 'DefaultTextFontName', 'Times New Roman');

%% 2. 加载 Python 导出的仿真结果数据
out_dir = '{matlab_out_dir}';   % 输出目录（图、.fig 全部落到此处）
load(fullfile(out_dir, 'data.mat'));
n_mu    = double(n_mu);
n_sigma = double(n_sigma);
sim_T   = double(simulation_times);
n0      = double(sample_size_n);
nk      = double(sample_size_N);
xi_mu    = 1:n_mu;
xi_sigma = 1:n_sigma;

%% 3. 归一化并插值（与 Python 端逻辑一致）
ST   = double(ST_disp);
half = max(ST(:))/2;
ST_n = (ST-half)/(max(abs(ST(:)-half))+1e-12);
[XI,YI] = meshgrid(linspace(1,n_mu,300),linspace(1,n_sigma,300));
ST_interp = interp2(xi_mu,xi_sigma,ST_n',XI,YI,'cubic');
ST_interp = max(-1,min(1,ST_interp));

%% 4. 热力图
figure('Units','centimeters','Position',[2 2 10 8]);
imagesc(xi_mu,xi_sigma,ST_interp);
set(gca,'YDir','normal');
colormap(gca,winter); clim([-1 1]);
cb = colorbar; cb.Ticks = -1:0.5:1;
hold on;
contour(linspace(1,n_mu,300),linspace(1,n_sigma,300),ST_interp,[0 0],'k','LineWidth',3);
set(gca,'XTick',xi_mu,   'XTickLabel',string(xi_mu),   'FontSize',10);
set(gca,'YTick',xi_sigma,'YTickLabel',string(xi_sigma),'FontSize',10);
xlabel('\\epsilon^\\mu',   'FontSize',13,'Interpreter','tex');
ylabel('\\epsilon^\\Sigma','FontSize',13,'Interpreter','tex');
title(sprintf('n_0 = %d, n_k = %d', n0, nk), 'Interpreter','tex','FontSize',12);
text(0.04,0.96,'(a)','Units','normalized','FontSize',13,...
     'FontWeight','bold','VerticalAlignment','top');
box on; axis tight;
exportgraphics(gcf, fullfile(out_dir,'heatmap_matlab.png'),'Resolution',300);
savefig(gcf, fullfile(out_dir,'heatmap_matlab.fig'));
disp('热力图已保存 (png + fig)');

%% 5. 3D 条形图
figure('Units','centimeters','Position',[14 2 12 10]);
ax3 = axes;
b = bar3(ST);
for k = 1:length(b)
    zdata = b(k).ZData; b(k).CData = zdata; b(k).FaceColor = 'interp';
end
colormap(ax3,winter); clim([0 sim_T]);
cb3 = colorbar; cb3.Label.String = 'Select Count'; cb3.FontSize = 9;
cb3.Ticks = 0:round(sim_T/8)*2:sim_T;
for i = 1:n_mu
    for j = 1:n_sigma
        if ST(i,j) > 0
            text(j,i,ST(i,j)+1.5,num2str(round(ST(i,j))),...
                 'HorizontalAlignment','center','VerticalAlignment','bottom',...
                 'FontSize',8,'Color','k');
        end
    end
end
set(ax3,'XTick',xi_sigma,'XTickLabel',string(xi_sigma),'FontSize',10);
set(ax3,'YTick',xi_mu,   'YTickLabel',string(xi_mu),   'FontSize',10);
xlabel('\\epsilon^\\Sigma','FontSize',12,'Interpreter','tex');
ylabel('\\epsilon^\\mu',   'FontSize',12,'Interpreter','tex');
zlabel('Select Count','FontSize',11);
zlim([0 sim_T+2]); zticks(0:round(sim_T/8)*2:sim_T);
title(sprintf('n_0 = %d, n_k = %d', n0, nk), 'Interpreter','tex','FontSize',12);
text(-0.08,1.03,'(e)','Units','normalized','FontSize',13,...
     'FontWeight','bold','VerticalAlignment','top');
view(-50,30); grid on;
exportgraphics(gcf, fullfile(out_dir,'3d_matlab.png'),'Resolution',300);
savefig(gcf, fullfile(out_dir,'3d_matlab.fig'));
disp('3D直方图已保存 (png + fig)');

%% 6. 组合图（热力图 + 3D 柱，1×2 布局）
figure('Units','centimeters','Position',[2 14 22 9]);
subplot(1,2,1);
imagesc(xi_mu,xi_sigma,ST_interp);
set(gca,'YDir','normal');
colormap(gca,winter); clim([-1 1]);
cb = colorbar; cb.Ticks=-1:0.5:1; cb.FontSize=9;
hold on;
contour(linspace(1,n_mu,300),linspace(1,n_sigma,300),ST_interp,[0 0],'k','LineWidth',3);
set(gca,'XTick',xi_mu,   'XTickLabel',string(xi_mu),   'FontSize',10);
set(gca,'YTick',xi_sigma,'YTickLabel',string(xi_sigma),'FontSize',10);
xlabel('\\epsilon^\\mu',   'FontSize',13,'Interpreter','tex');
ylabel('\\epsilon^\\Sigma','FontSize',13,'Interpreter','tex');
title(sprintf('n_0 = %d, n_k = %d', n0, nk), 'Interpreter','tex','FontSize',12);
text(0.04,0.96,'(a)','Units','normalized','FontSize',13,...
     'FontWeight','bold','VerticalAlignment','top');
box on; axis tight;
subplot(1,2,2); ax3b = gca;
b2 = bar3(ST);
for k = 1:length(b2)
    zdata = b2(k).ZData; b2(k).CData=zdata; b2(k).FaceColor='interp';
end
colormap(ax3b,winter); clim([0 sim_T]);
cb3b = colorbar; cb3b.Label.String='Select Count'; cb3b.FontSize=9;
for i = 1:n_mu
    for j = 1:n_sigma
        if ST(i,j) > 0
            text(j,i,ST(i,j)+1.5,num2str(round(ST(i,j))),...
                 'HorizontalAlignment','center','VerticalAlignment','bottom',...
                 'FontSize',8,'Color','k');
        end
    end
end
set(ax3b,'XTick',xi_sigma,'XTickLabel',string(xi_sigma),'FontSize',10);
set(ax3b,'YTick',xi_mu,   'YTickLabel',string(xi_mu),   'FontSize',10);
xlabel('\\epsilon^\\Sigma','FontSize',12,'Interpreter','tex');
ylabel('\\epsilon^\\mu',   'FontSize',12,'Interpreter','tex');
zlabel('Select Count','FontSize',11);
zlim([0 sim_T+2]); grid on; view(-50,30);
title(sprintf('n_0 = %d, n_k = %d', n0, nk), 'Interpreter','tex','FontSize',12);
text(-0.08,1.03,'(e)','Units','normalized','FontSize',13,...
     'FontWeight','bold','VerticalAlignment','top');
exportgraphics(gcf, fullfile(out_dir,'combo_matlab.png'),'Resolution',300);
savefig(gcf, fullfile(out_dir,'combo_matlab.fig'));
disp('组合图已保存 (png + fig)');
disp('全部完成！');
"""

    # ── 步骤3：将脚本写入 .m 文件 ─────────────────────────────────────────────
    m_path = os.path.join(code_dir, 'plot.m')
    with open(m_path, 'w', encoding='utf-8') as f:
        f.write(m_script)
    print(f'[MATLAB] 绘图脚本已生成: {m_path}')

    # ── 步骤4：自动查找并调用 MATLAB ─────────────────────────────────────────
    # 优先查找 /Applications/MATLAB_R*.app（macOS 标准安装路径，按版本号倒序取最新）
    # 次优使用 PATH 中的 matlab（Linux/Windows 集群环境）
    import subprocess, glob, shutil
    _matlab_bin = None
    for _p in (sorted(glob.glob('/Applications/MATLAB_R*.app/bin/matlab'), reverse=True)
               + [shutil.which('matlab')]):
        if _p and os.path.isfile(_p):
            _matlab_bin = _p
            break

    if _matlab_bin:
        print(f'[MATLAB] 使用: {_matlab_bin}')
        print('[MATLAB] 正在执行绘图脚本，请稍候...')
        # -nodisplay -nosplash -nodesktop：无头模式运行（不弹出 GUI 界面）
        # -batch "run('...')": 执行指定 .m 脚本后自动退出
        ret = subprocess.run(
            [_matlab_bin, '-nodisplay', '-nosplash', '-nodesktop',
             '-batch', f"run('{m_path}');"],
            capture_output=True, text=True, timeout=300   # 超时 5 分钟
        )
        if ret.returncode == 0:
            print('[MATLAB] 绘图完成！')
        else:
            print(f'[MATLAB] 执行出错, stderr:\n{ret.stderr}')
            if ret.stdout:
                print(f'[MATLAB] stdout:\n{ret.stdout}')
    else:
        print('[MATLAB] 未找到 MATLAB，跳过自动执行。请手动运行:', m_path)


# =============================================================================
#  对外接口 1：单次结果绘图
# =============================================================================

def plot_selection(select_times_pro, h_mu, h_sigma,
                   which_Exm, sample_size_n, sample_size_N,
                   simulation_times, code_dir=None):
    """
    为单次（单个 n）的仿真结果生成热力图、3D 条形图、组合图，并调用 MATLAB 输出高质量版本。

    本函数是本模块的主要对外接口，由 MstMdsp_simulation_main.py 第5节调用。

    所有产物落到专属目录：
        outputs/exm{E}_n{n}_N{N}_T{T}/
            heatmap.png         热力图 (Python, dpi=150)
            3d.png              3D 条形图 (Python, dpi=150)
            combo.png           热力图+3D 组合 (Python, dpi=150)
            data.mat            MATLAB 数据
            plot.m              MATLAB 绘图脚本
            heatmap_matlab.png  热力图 (MATLAB, dpi=300, 论文质量)
            3d_matlab.png       3D 条形图 (MATLAB, dpi=300)
            combo_matlab.png    组合图 (MATLAB, dpi=300)
            hist.png            实验1/3 的 1D 条形图（替代上述）

    Parameters
    ----------
    select_times_pro : array-like, shape (n_mu, n_sigma) or None
        各数据源被选中的次数矩阵。
        行索引对应 h_mu 中的 ε^μ 取值，列索引对应 h_sigma 中的 ε^Σ 取值。
        若为 None，则跳过可视化并打印警告。
    h_mu : list of float
        均值扰动参数 ε^μ 的取值列表（含负值和正值，以0为中心分布）。
        例如：[-3, -1.5, -0.5, 0, 0.5, 1.5, 3]
    h_sigma : list of float
        方差扰动参数 ε^Σ 的取值列表（格式同 h_mu）。
    which_Exm : int
        仿真实验编号（1=单源线性，2=多源异质逻辑，3=...）。
        影响图的布局：which_Exm in {1,3} 生成1D条形图；which_Exm==2 生成热力图+3D柱。
    sample_size_n : int
        标记样本数，用于图标题（如 '$n_0=250$'）和文件命名。
    sample_size_N : int
        无标签样本数，用于图标题（如 '$n_k=5000$'）。
    simulation_times : int
        蒙特卡洛重复次数，用于设定颜色轴上限和 z 轴范围。
    code_dir : str or None
        输出根目录覆盖入口；None 时使用模块默认 _OUTPUTS_DIR（= code/outputs）。
        实际产物会落到 <code_dir>/exm{E}_n{n}_N{N}_T{T}/ 子目录里。
    """
    if select_times_pro is None:
        print('[SelectionViz] select_times_pro 为 None，跳过可视化')
        return

    # ── 输出目录：每次实验一个专属文件夹 outputs/exm{E}_n{n}_N{N}_T{T}/ ─────
    # code_dir 此处用作"输出根目录"的覆盖入口（极少需要），默认走 _OUTPUTS_DIR
    outputs_root = code_dir if code_dir is not None else _OUTPUTS_DIR
    run_dir = _make_run_dir(which_Exm, sample_size_n, sample_size_N,
                            simulation_times, outputs_root=outputs_root)
    print(f'[SelectionViz] 本次实验输出目录: {run_dir}')

    # ── 数据预处理（所有绘图函数共享同一份数据包）────────────────────────────
    D     = _prep_data(select_times_pro, h_mu, h_sigma, simulation_times)
    title = f'$n_0 = {sample_size_n},\\ n_k = {sample_size_N}$'  # mathtext: 下标 + 强制空格

    if which_Exm in (1, 3):
        # ── 实验1/3：单一 ε^μ 轴，绘制简单1D条形图 ──────────────────────────
        # 取矩阵对角线（若方阵）或第一列作为频次
        fig, ax = plt.subplots(figsize=(8, 5))
        x_pos       = D['xi_mu']
        diag_counts = (np.diag(D['ST_disp'])
                       if D['ST_disp'].shape[0] == D['ST_disp'].shape[1]
                       else D['ST_disp'][:, 0])
        ax.bar(x_pos, diag_counts, color='#2b8cbe', edgecolor='white', width=0.55)
        ax.set_xticks(x_pos); ax.set_xticklabels(D['mu_labels'], fontsize=9)
        ax.set_xlabel('Unlabeled Dataset', fontsize=13)
        ax.set_ylabel('Select Count',      fontsize=13)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, simulation_times + 5)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        out = os.path.join(run_dir, 'hist.png')
        plt.savefig(out, dpi=150)
        plt.close()
        print(f'[绘图] 条形图已保存: {out}')

    else:  # which_Exm == 2：双参数（ε^μ, ε^Σ）网格，绘制热力图 + 3D 柱
        # ── 单独热力图 ─────────────────────────────────────────────────────
        fig_h, ax_h = plt.subplots(figsize=(5.5, 4.5))
        _draw_heatmap(ax_h, fig_h, D, title, panel_label='(a)')
        plt.tight_layout()
        out_h = os.path.join(run_dir, 'heatmap.png')
        plt.savefig(out_h, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'[绘图] 热力图已保存: {out_h}')

        # ── 单独 3D 条形图 ─────────────────────────────────────────────────
        fig_3d = plt.figure(figsize=(7, 6))
        ax_3d  = fig_3d.add_subplot(111, projection='3d')
        _draw_3d_bars(ax_3d, D, title, simulation_times, panel_label='(e)')
        # colorbar 需从 ScalarMappable 手动创建（3D axes 不支持 imshow 的自动 colorbar）
        cbar3 = fig_3d.colorbar(D['sm'], ax=ax_3d, shrink=0.45, aspect=12, pad=0.06)
        cbar3.set_label('Select Count', fontsize=9)
        cbar3.set_ticks(D['zticks'])
        cbar3.ax.tick_params(labelsize=8)
        plt.tight_layout()
        out_3d = os.path.join(run_dir, '3d.png')
        plt.savefig(out_3d, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'[绘图] 3D 条形图已保存: {out_3d}')

        # ── 组合图（热力图 + 3D 柱，1×2 布局）─────────────────────────────
        fig_c = plt.figure(figsize=(13, 5.5))
        ax_l  = fig_c.add_subplot(1, 2, 1)
        _draw_heatmap(ax_l, fig_c, D, title, panel_label='(a)')
        ax_r  = fig_c.add_subplot(1, 2, 2, projection='3d')
        _draw_3d_bars(ax_r, D, title, simulation_times, panel_label='(e)')
        cbar_c = fig_c.colorbar(D['sm'], ax=ax_r, shrink=0.45, aspect=12, pad=0.06)
        cbar_c.set_label('Select Count', fontsize=9)
        cbar_c.set_ticks(D['zticks'])
        cbar_c.ax.tick_params(labelsize=8)
        plt.subplots_adjust(wspace=0.3)
        out_c = os.path.join(run_dir, 'combo.png')
        plt.savefig(out_c, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'[绘图] 组合图已保存: {out_c}')

    # ── 导出 MATLAB 高质量版（无论 which_Exm 为何值均执行）──────────────────
    # 注意：此处把 run_dir 作为输出目录传给 _export_matlab，
    #       使 .mat/.m/MATLAB-png 全部落到本次实验专属文件夹内。
    _export_matlab(D, which_Exm, sample_size_n, sample_size_N,
                   simulation_times, run_dir)


# =============================================================================
#  对外接口 2：多 n 值论文大图（2 行 × N 列）
# =============================================================================

def plot_multi_panel(results_dict, which_Exm, sample_size_N, simulation_times,
                     code_dir=None, export_matlab=False):
    """
    生成论文用多面板大图，将多个样本量 n 的结果并排展示。

    图片布局：
      · 上行（第1行）：各 n 值对应的热力图
      · 下行（第2行）：各 n 值对应的 3D 条形图
      · 列数 = len(results_dict)，每列对应一个样本量 n

    面板标签：
      · 上行：(a), (b), (c), (d), ...
      · 下行：(e), (f), (g), (h), ...

    Parameters
    ----------
    results_dict : dict, {sample_size_n: (select_times_pro, h_mu, h_sigma)}
        键 = 样本量 n（int），值 = 三元组：
          · select_times_pro : array-like (n_mu, n_sigma) — 选中次数矩阵
          · h_mu             : list of float — ε^μ 取值列表
          · h_sigma          : list of float — ε^Σ 取值列表
        字典中的键将按从小到大排序，依次作为图的各列。
    which_Exm : int
        实验编号（用于输出文件命名）。
    sample_size_N : int
        无标签样本数（用于图标题）。
    simulation_times : int
        蒙特卡洛重复次数（用于颜色轴范围和 z 轴上限）。
    code_dir : str or None
        输出目录路径。None 时默认使用本文件所在目录（_CODE_DIR）。
    export_matlab : bool
        是否同时生成 MATLAB 绘图脚本并调用 MATLAB 导出高质量图片。默认 False，
        避免主入口运行时被外部 MATLAB 进程阻塞。

    Returns
    -------
    out_big : str
        生成的多面板大图的完整文件路径。

    Example
    -------
    from SelectionViz import plot_multi_panel

    # 先分别运行主脚本得到四个 n 值的结果，再汇总：
    results = {
        250:  (stp_250,  h_mu, h_sigma),
        500:  (stp_500,  h_mu, h_sigma),
        1000: (stp_1000, h_mu, h_sigma),
        2000: (stp_2000, h_mu, h_sigma),
    }
    plot_multi_panel(results, which_Exm=2, sample_size_N=5000, simulation_times=100)
    """
    # ── 多面板大图统一保存到 outputs/multi_panel/ 子目录 ───────────────────────
    outputs_root = code_dir if code_dir is not None else _OUTPUTS_DIR
    panel_dir = os.path.join(outputs_root, 'multi_panel')
    os.makedirs(panel_dir, exist_ok=True)

    # 按样本量从小到大排列，确保图列顺序一致
    n_list = sorted(results_dict.keys())
    n_cols = len(n_list)

    # 面板标签：上行 (a)(b)(c)(d)，下行 (e)(f)(g)(h)
    panel_top = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)'][:n_cols]
    panel_bot = ['(e)', '(f)', '(g)', '(h)', '(i)', '(j)'][:n_cols]

    # 整体图形大小：宽度按列数等比例放大，高度固定两行
    fig = plt.figure(figsize=(5.5 * n_cols, 11))

    for col_idx, n in enumerate(n_list):
        stp, h_mu, h_sigma = results_dict[n]
        D     = _prep_data(stp, h_mu, h_sigma, simulation_times)
        title = f'$n_0 = {n},\\ n_k = {sample_size_N}$'

        # ── 上行：热力图（位于第1行，第 col_idx+1 列）────────────────────────
        ax_h = fig.add_subplot(2, n_cols, col_idx + 1)
        _draw_heatmap(ax_h, fig, D, title, panel_label=panel_top[col_idx])

        # ── 下行：3D 柱状图（位于第2行，第 col_idx+1 列）─────────────────────
        ax_3 = fig.add_subplot(2, n_cols, n_cols + col_idx + 1, projection='3d')
        _draw_3d_bars(ax_3, D, title, simulation_times,
                      panel_label=panel_bot[col_idx])
        # 每个 3D 子图单独添加 colorbar
        cbar = fig.colorbar(D['sm'], ax=ax_3, shrink=0.45, aspect=12, pad=0.06)
        cbar.set_label('Select Count', fontsize=8)
        cbar.set_ticks(D['zticks'])
        cbar.ax.tick_params(labelsize=7)

    # 调整子图间距（水平间距稍大以容纳 colorbar）
    plt.subplots_adjust(wspace=0.38, hspace=0.22)

    # 输出文件名包含所有 n 值，便于区分不同配置的大图
    ns_str  = '_'.join(str(n) for n in n_list)
    out_big = os.path.join(panel_dir,
                           f'panel_exm{which_Exm}_nk{sample_size_N}_n{ns_str}.png')
    plt.savefig(out_big, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'[绘图] 多面板大图已保存: {out_big}')

    # ── 可选：导出 MATLAB 高质量多面板大图 ────────────────────────────────────────
    if export_matlab:
        _export_matlab_multi_panel(results_dict, which_Exm, sample_size_N,
                                   simulation_times, panel_dir)

    return out_big


def _export_matlab_multi_panel(results_dict, which_Exm, sample_size_N,
                                simulation_times, out_dir):
    """
    将多个 n 值的选择频次矩阵导出为多个 .mat 文件，并生成 MATLAB 脚本，
    绘制与论文图一致的 2行×N列 多面板大图（上行热力图，下行3D柱）。

    产物（均落到 out_dir）：
      · data_n{n}.mat                   — 各 n 值的数据文件
      · plot_multi_panel.m              — MATLAB 绘图脚本
      · multi_panel_matlab.png          — MATLAB 生成的多面板大图（300 dpi）
      · multi_panel_matlab.fig          — 可在 MATLAB 中继续编辑的 .fig 文件

    Parameters
    ----------
    results_dict : dict, {sample_size_n: (select_times_pro, h_mu, h_sigma)}
    which_Exm    : int
    sample_size_N: int
    simulation_times : int
    out_dir      : str  — 输出目录（由 plot_multi_panel 传入 panel_dir）
    """
    import scipy.io as sio

    n_list = sorted(results_dict.keys())
    n_cols = len(n_list)

    # ── 步骤1：为每个 n 值保存 .mat 数据文件 ──────────────────────────────────
    for n in n_list:
        stp, h_mu, h_sigma = results_dict[n]
        D = _prep_data(stp, h_mu, h_sigma, simulation_times)
        mat_data = {
            'ST_disp':          D['ST_disp'],
            'h_mu_disp':        D['h_mu_disp'].reshape(-1, 1),
            'h_sigma_disp':     D['h_sigma_disp'].reshape(-1, 1),
            'n_mu':             float(D['n_mu']),
            'n_sigma':          float(D['n_sigma']),
            'simulation_times': float(simulation_times),
            'sample_size_n':    float(n),
            'sample_size_N':    float(sample_size_N),
        }
        mat_path = os.path.join(out_dir, f'data_n{n}.mat')
        sio.savemat(mat_path, mat_data)
        print(f'[MATLAB多面板] 数据已保存: {mat_path}')

    matlab_out_dir = os.path.abspath(out_dir).replace("'", "''")

    # ── 步骤2：生成 MATLAB 绘图脚本 ──────────────────────────────────────────
    # 面板标签：上行 (a)(b)(c)(d)，下行 (e)(f)(g)(h)
    top_labels = list('abcdefghij')[:n_cols]
    bot_labels = list('abcdefghij')[n_cols:2*n_cols]

    # 构造每列对应的 n 值列表（MATLAB 数组格式）
    n_vals_str = '[' + ' '.join(str(n) for n in n_list) + ']'

    # 构造 n_list 的 MATLAB cell 数组（用于按列加载不同 .mat 文件）
    n_cell_str = '{' + ', '.join(str(n) for n in n_list) + '}'

    top_label_cell = '{' + ', '.join(f"'({l})'" for l in top_labels) + '}'
    bot_label_cell = '{' + ', '.join(f"'({l})'" for l in bot_labels) + '}'

    # 子图宽度（cm）和整体图宽
    col_w   = 10          # 每列约10 cm（与单图一致）
    fig_w   = col_w * n_cols + 4   # 额外留给 colorbar 的空间
    fig_h   = 20          # 两行高度约20 cm

    m_script = f"""% =========================================================
%  MATLAB 多面板绘图脚本 — 2行×{n_cols}列 选择频次大图
%  由 SelectionViz.py (_export_matlab_multi_panel) 自动生成
%  上行：热力图；下行：3D条形图
% =========================================================
clear; close all; clc;

%% 全局字体
set(0, 'DefaultAxesFontName', 'Times New Roman');
set(0, 'DefaultTextFontName', 'Times New Roman');

out_dir   = '{matlab_out_dir}';
n_vals    = {n_vals_str};
n_cols    = {n_cols};
sim_T     = {simulation_times};
nk        = {sample_size_N};
top_labels = {top_label_cell};
bot_labels = {bot_label_cell};

fig = figure('Units','centimeters','Position',[1 1 {fig_w} {fig_h}]);

for ci = 1:n_cols
    n0 = n_vals(ci);
    load(fullfile(out_dir, sprintf('data_n%d.mat', n0)));
    n_mu    = double(n_mu);
    n_sigma = double(n_sigma);
    ST      = double(ST_disp);
    xi_mu    = 1:n_mu;
    xi_sigma = 1:n_sigma;
    title_str = sprintf('n_0 = %d, n_k = %d', n0, nk);

    %% 归一化 + 插值
    half     = max(ST(:))/2;
    ST_n     = (ST - half) / (max(abs(ST(:)-half)) + 1e-12);
    [XI,YI]  = meshgrid(linspace(1,n_mu,300), linspace(1,n_sigma,300));
    ST_interp = interp2(xi_mu, xi_sigma, ST_n', XI, YI, 'cubic');
    ST_interp = max(-1, min(1, ST_interp));

    %% 上行：热力图
    ax_h = subplot(2, n_cols, ci);
    imagesc(xi_mu, xi_sigma, ST_interp);
    set(ax_h, 'YDir','normal');
    colormap(ax_h, winter); clim([-1 1]);
    cb = colorbar(ax_h); cb.Ticks = -1:0.5:1; cb.FontSize = 8;
    hold(ax_h,'on');
    contour(ax_h, linspace(1,n_mu,300), linspace(1,n_sigma,300), ...
            ST_interp, [0 0], 'k', 'LineWidth', 3);
    set(ax_h,'XTick',xi_mu,    'XTickLabel',string(xi_mu),    'FontSize',10);
    set(ax_h,'YTick',xi_sigma, 'YTickLabel',string(xi_sigma), 'FontSize',10);
    xlabel(ax_h, '\\epsilon^\\mu',    'FontSize',13,'Interpreter','tex');
    ylabel(ax_h, '\\epsilon^\\Sigma', 'FontSize',13,'Interpreter','tex');
    title(ax_h, title_str, 'Interpreter','tex','FontSize',11,'FontWeight','bold');
    text(ax_h, 0.04, 0.96, top_labels{{ci}}, 'Units','normalized', ...
         'FontSize',13,'FontWeight','bold','VerticalAlignment','top');
    box(ax_h,'on'); axis(ax_h,'tight');

    %% 下行：3D条形图
    ax3 = subplot(2, n_cols, n_cols + ci);
    b = bar3(ax3, ST);
    for k = 1:length(b)
        zdata = b(k).ZData; b(k).CData = zdata; b(k).FaceColor = 'interp';
    end
    colormap(ax3, winter); clim(ax3, [0 sim_T]);
    cb3 = colorbar(ax3); cb3.Label.String='Select Count'; cb3.FontSize=8;
    cb3.Ticks = 0:round(sim_T/8)*2:sim_T;
    hold(ax3,'on');
    for i = 1:n_mu
        for j = 1:n_sigma
            if ST(i,j) > 0
                text(ax3, j, i, ST(i,j)+1.5, num2str(round(ST(i,j))), ...
                     'HorizontalAlignment','center','VerticalAlignment','bottom', ...
                     'FontSize',7,'Color','k');
            end
        end
    end
    set(ax3,'XTick',xi_sigma,'XTickLabel',string(xi_sigma),'FontSize',9);
    set(ax3,'YTick',xi_mu,   'YTickLabel',string(xi_mu),   'FontSize',9);
    xlabel(ax3,'\\epsilon^\\Sigma','FontSize',11,'Interpreter','tex');
    ylabel(ax3,'\\epsilon^\\mu',   'FontSize',11,'Interpreter','tex');
    zlabel(ax3,'Select Count','FontSize',10);
    zlim(ax3,[0 sim_T+2]);
    zticks(ax3, 0:round(sim_T/8)*2:sim_T);
    title(ax3, title_str,'Interpreter','tex','FontSize',11,'FontWeight','bold');
    text(ax3,-0.08,1.03, bot_labels{{ci}}, 'Units','normalized', ...
         'FontSize',13,'FontWeight','bold','VerticalAlignment','top');
    view(ax3,-50,30); grid(ax3,'on');
end

%% 调整间距并保存
set(fig,'Units','normalized');
set(fig,'Units','centimeters');
exportgraphics(fig, fullfile(out_dir,'multi_panel_matlab.png'),'Resolution',300);
savefig(fig, fullfile(out_dir,'multi_panel_matlab.fig'));
disp('多面板大图已保存 (png + fig)');
"""

    m_path = os.path.join(out_dir, 'plot_multi_panel.m')
    with open(m_path, 'w', encoding='utf-8') as f:
        f.write(m_script)
    print(f'[MATLAB多面板] 绘图脚本已生成: {m_path}')

    # ── 步骤3：自动调用 MATLAB 执行 ──────────────────────────────────────────
    import subprocess, glob, shutil
    _matlab_bin = None
    for _p in (sorted(glob.glob('/Applications/MATLAB_R*.app/bin/matlab'), reverse=True)
               + [shutil.which('matlab')]):
        if _p and os.path.isfile(_p):
            _matlab_bin = _p
            break

    if _matlab_bin:
        print(f'[MATLAB多面板] 使用: {_matlab_bin}')
        print('[MATLAB多面板] 正在执行，请稍候...')
        ret = subprocess.run(
            [_matlab_bin, '-nodisplay', '-nosplash', '-nodesktop',
             '-batch', f"run('{m_path}');"],
            capture_output=True, text=True, timeout=600
        )
        if ret.returncode == 0:
            print('[MATLAB多面板] 多面板大图绘制完成！')
        else:
            print(f'[MATLAB多面板] 执行出错, stderr:\n{ret.stderr}')
            if ret.stdout:
                print(f'[MATLAB多面板] stdout:\n{ret.stdout}')
    else:
        print('[MATLAB多面板] 未找到 MATLAB，请手动运行:', m_path)
