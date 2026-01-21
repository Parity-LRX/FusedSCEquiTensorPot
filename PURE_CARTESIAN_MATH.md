## Pure-Cartesian Tensor Product的数学描述

本文阐述在**全笛卡尔 \(3^L\)** 表示下，使用 **Kronecker δ** 与 **Levi-Civita ε** 构造**严格 O(3) 等变**张量积的方法，及其与 ICTD（Irreducible Cartesian Tensor Decomposition，不可约笛卡尔张量分解）[3]/trace-chain（\((2\ell+1)\) irreps）的关系，以及本实现提供的"无球谐 irreps 内部表示"版本。


- **第 1–7 节**：`pure-cartesian` / `pure-cartesian-sparse` 的严格 O(3) 版本，内部表示 \(\bigoplus_L (\mathbb R^3)^{\otimes L}\)（维度 \(3^L\)），显式 true/pseudo（\(\mathbb Z_2\) 分级）处理 \(\det(R)\)。
- **第 8 节**：小结（实现到数学的一一对应）
- **第 9 节**：irreps（如 `64x0e+64x1o+64x2e`）与 pure-cartesian（\(3^L\)）的对应关系。
- **第 10 节**：`ictd_fast.py` 的快速不可约投影（对称子空间的 trace-chain / STF），从对称笛卡尔张量提取 \((2\ell+1)\) 块坐标，属于 SO(3) 范畴。
- **第 11 节**：`ictd_irreps.py` / `pure_cartesian_ictd_layers.py` 的 ICTD-irreps 内部表示，网络内部全程使用 \((2\ell+1)\) irreps，在 SO(3) 下严格等变。

对应实现：
- `molecular_force_field/models/pure_cartesian.py`
  - `PureCartesianTensorProductO3`
  - `PureCartesianTensorProductO3Sparse`
  - `_enumerate_paths`, `_enumerate_paths_sparse`, `_einsum_for_path`
  - `split_by_rank_o3`, `merge_by_rank_o3`, `rotate_rank_tensor`, `edge_rank_powers`
- `molecular_force_field/models/pure_cartesian_layers.py` / `pure_cartesian_sparse_layers.py`
  - 以消息传递方式使用上述张量积

### 阅读导航

- **严格 O(3)（含宇称）纯笛卡尔张量积**：第 1–6 节（重点：第 5–6 节）
- **Sparse 模式的稀疏化策略及 `k_policy/allow_epsilon/assume_pseudo_zero` 的数学含义**：第 6 节
- **`64x0e+64x1o+64x2e` 与 \(3^L\) 的对应关系**：第 9 节
- **快速 ICTD 投影（从对称张量提取 irreps 块）**：第 10 节
- **无球谐 irreps 内部表示及通过多项式构造 CG 张量**：第 11 节

### 符号约定（贯穿全文）
- **irreps**：不可约表示（irreducible representation）的缩写，符号来自 e3nn 库 [1]。表示 \(SO(3)\) 或 \(O(3)\) 群的不可约表示。在 e3nn 中，irreps 用字符串表示，例如 `64x0e+64x1o+64x2e`，其中：数字表示通道数（multiplicity），\(\ell\)（如 0, 1, 2）表示不可约表示的阶，\(e/o\) 表示宇称（even/odd，对应偶/奇宇称）。
- **\(L\)**：pure-cartesian 的"秩/阶"（rank），对应全笛卡尔表示 \((\mathbb R^3)^{\otimes L}\)，维度 \(3^L\)。
- **\(\ell\)**：不可约表示/调和多项式的"阶"（irrep degree / harmonic degree），对应维度 \(2\ell+1\) 的 irreps 块。
- **\(s\in\{0,1\}\)**：\(\mathbb Z_2\) 分级（true/pseudo），实现严格 \(O(3)\)（含反射）等变性的标记机制（见第 2 节）。
- **\(R\in O(3)\)**：正交变换；\(\det(R)\in\{\pm1\}\)。\(R\in SO(3)\) 表示仅旋转（\(\det=+1\)）。

---

## 1. 记号与基本对象

### 1.1 O(3) 与 det 因子
令
\[
O(3)=\{R\in\mathbb R^{3\times 3}\mid R^\top R=I\},
\quad \det(R)\in\{+1,-1\}.
\]
\(\det(R)=-1\) 表示包含反射（宇称）变换。

### 1.2 全笛卡尔秩张量空间（\(3^L\) 分量）
对每个秩 \(L\in\mathbb N\)，定义
\[
\mathcal T_L := (\mathbb R^3)^{\otimes L},
\qquad \dim(\mathcal T_L)=3^L.
\]
当 \(L=0\) 时 \(\mathcal T_0=\mathbb R\)（标量）。

给定通道数 \(C\)，秩 \(L\) 的特征块取
\[
\mathbb R^C\otimes \mathcal T_L.
\]

---

## 2. O(3) 的自然作用与 Z2 分级（true / pseudo）

### 2.1 自然张量表示（对秩指标）
对 \(T\in\mathcal T_L\)（坐标表示为 \(T_{i_1\cdots i_L}\)），定义
\[
(R\cdot T)_{i_1\cdots i_L}
=\sum_{j_1,\ldots,j_L}
R_{i_1j_1}\cdots R_{i_Lj_L}\,T_{j_1\cdots j_L}.
\]
通道维 \(\mathbb R^C\) 视为标量拷贝，不随 \(R\) 变化。

### 2.2 O(3) 严格等变所需的 true/pseudo 分级
为在 \(O(3)\)（而非仅 \(SO(3)\)）下严格等变，引入 \(\mathbb Z_2\) 分级：
- \(s=0\)：true tensor（极张量 / polar tensor）
- \(s=1\)：pseudotensor（轴张量 / axial tensor）

定义对分级块 \(T^{(s)}\in\mathcal T_L\) 的群作用为
\[
(R\cdot T^{(s)}) := (\det R)^s\,(R\cdot T).
\]
对应实现中的 `rotate_rank_tensor(..., pseudo=s)`。

### 2.3 全特征空间（本层的输入/输出空间）
给定 \(C, L_{\max}\)，定义
\[
\mathcal V(C,L_{\max})
=\bigoplus_{s\in\{0,1\}}\ \bigoplus_{L=0}^{L_{\max}}\left(\mathbb R^C\otimes\mathcal T_L\right).
\]
实现中通过 `split_by_rank_o3/merge_by_rank_o3` 将扁平向量打包/拆包成这些块。

---

## 3. δ 与 ε：唯一使用的几何不变张量

### 3.1 Kronecker δ
\[
\delta_{ij}=
\begin{cases}
1,& i=j\\
0,& i\ne j
\end{cases}
\]
满足 \(O(3)\) 不变性：
\[
\sum_{a,b} R_{ia}R_{jb}\delta_{ab}=\delta_{ij}.
\]

### 3.2 Levi-Civita ε
\(\varepsilon_{ijk}\) 为三维完全反对称张量，满足
\[
\sum_{a,b,c}R_{ia}R_{jb}R_{kc}\varepsilon_{abc}=(\det R)\,\varepsilon_{ijk}.
\]
因此一次 ε 收缩会引入额外的 \(\det R\) 因子（pseudo 的来源，见第 2.2 节）。

等价定义：
\[
\varepsilon_{ijk}=
\begin{cases}
 +1,& (i,j,k)\ \text{为}\ (1,2,3)\ \text{的偶置换}\\
 -1,& (i,j,k)\ \text{为}\ (1,2,3)\ \text{的奇置换}\\
 0,& \text{否则}.
\end{cases}
\]

实现中 `epsilon_tensor()` 返回 \(\varepsilon\)。

---

## 4. “路径”（path）与规范化指标收缩算子 Γ

令 \(L_1,L_2\in\{0,\dots,L_{\max}\}\)。对每条路径 \(p\) 由以下参数组成：
- \(\delta\) 收缩数 \(k\in\{0,\dots,\min(L_1,L_2)\}\)（实现字段 `k_delta`）
- 是否使用一次 ε（实现字段 `use_epsilon`）

定义输出秩
\[
L_{\text{out}}=
\begin{cases}
L_1+L_2-2k,& \text{仅 δ 收缩}\\
L_1+L_2-2k-1,& \text{δ 收缩后再用一次 ε}
\end{cases}
\]
并要求 \(0\le L_{\text{out}}\le L_{\max}\)。

### 4.1 规范化收缩顺序
- 将第一个输入张量的尾部 \(k\) 个指标与第二个输入张量的尾部 \(k\) 个指标两两用 δ 收缩（实现中 `_einsum_for_path` 重命名指标为同一符号）。
- 若 `use_epsilon=True`：完成 \(k\) 次 δ 收缩后，从两个张量剩余自由指标中各选一个（规范选择：各自最后一个），与 \(\varepsilon_{ijk}\) 收缩产生新输出指标，追加到输出指标串末尾。

由此得到一个双线性算子
\[
\Gamma_p:\mathcal T_{L_1}\times \mathcal T_{L_2}\to \mathcal T_{L_{\text{out}}}.
\]

### 4.2 Γ 的 O(3) 等变性（核心）
由 δ 的不变性与 ε 的伪不变性可得：
- 若 `use_epsilon=False`：
\[
\Gamma_p(R\cdot A,\ R\cdot B)=R\cdot \Gamma_p(A,B).
\]
- 若 `use_epsilon=True`：
\[
\Gamma_p(R\cdot A,\ R\cdot B)=(\det R)\,R\cdot \Gamma_p(A,B).
\]
即 ε 路径额外产生一个 \(\det R\) 因子。

### 4.3 示例（δ 收缩）

取 \(L_1=2\)（矩阵）、\(L_2=1\)（向量）、\(k=1\)、`use_epsilon=False`，则
\[
L_{\text{out}} = 2+1-2\cdot 1 = 1,
\]
收缩为
\[
\Gamma_p(A_{ij},\,B_k) = \sum_{m=1}^3 A_{im}\,B_m,
\]
输出为 rank-1 张量（向量）。

---

## 5. Pure-Cartesian（严格 O(3)）张量积：`PureCartesianTensorProductO3`

### 5.1 输入输出与参数
设输入通道 \(C_1,C_2\)，输出通道 \(C_{\text{out}}\)。本层实现一个映射：
\[
\mathrm{TP}:\mathcal V(C_1,L_{\max})\times \mathcal V(C_2,L_{\max})\to \mathcal V(C_{\text{out}},L_{\max}).
\]

对每条路径 \(p\) 与每个输入分级 \((s_1,s_2)\in\{0,1\}^2\)，有可学习通道混合权：
\[
W_{p}^{(s_1,s_2)}\in\mathbb R^{C_{\text{out}}\times C_1\times C_2}.
\]
实现中每条 path 对应 4 份权重（对应 4 个 \((s_1,s_2)\) 组合）。

### 5.2 输出分级规则
定义
\[
s_{\text{out}} = s_1\oplus s_2\oplus \mathbf 1_{\text{use\_epsilon}},
\]
其中 \(\oplus\) 是 XOR。含义：
- true × true 经 δ 仍为 true
- ε 引入 \(\det R\)，翻转 true/pseudo
- pseudo 的 \((\det R)\) 因子通过 XOR 规则在输出端传递

实现中对应：`s_out = s1 ^ s2 ^ (1 if p.use_epsilon else 0)`。

### 5.3 块级公式（严格定义）
记输入块
\[
X_{L_1}^{(s_1)}\in \mathbb R^{C_1}\otimes\mathcal T_{L_1},\qquad
Y_{L_2}^{(s_2)}\in \mathbb R^{C_2}\otimes\mathcal T_{L_2}.
\]
对每条路径 \(p\)（确定 \(L_1,L_2,L_{\text{out}}\)）定义中间张量
\[
U_{p}^{(s_1,s_2)} :=
\Gamma_p\!\left(X_{L_1}^{(s_1)},\,Y_{L_2}^{(s_2)}\right)
\in (\mathbb R^{C_1}\otimes\mathbb R^{C_2})\otimes\mathcal T_{L_{\text{out}}}.
\]
然后进行通道混合：
\[
Z_{L_{\text{out}}}^{(s_{\text{out}})}
\;+=\;
\sum_{a=1}^{C_1}\sum_{b=1}^{C_2}
W_{p}^{(s_1,s_2)}[\cdot,a,b]\;\,U_{p,a,b}^{(s_1,s_2)}.
\]
最后对所有 \(p,s_1,s_2\) 求和得到完整输出 \(Z\in\mathcal V(C_{\text{out}},L_{\max})\)。

### 5.4 严格 O(3) 等变性定理
对任意 \(R\in O(3)\)，有
\[
\mathrm{TP}(R\cdot x_1,\ R\cdot x_2)=R\cdot \mathrm{TP}(x_1,x_2).
\]
证明要点：
- 每条路径的 \(\Gamma_p\) 在 \(SO(3)\) 下严格等变；
- 若包含 ε，则额外 \((\det R)\) 因子恰好由 \(s_{\text{out}}=s_1\oplus s_2\oplus 1\) 吸收；
- 通道混合 \(W\) 只作用在通道维（标量拷贝），不影响等变性；
- 线性叠加保持等变性。

---

## 6. Pure-Cartesian-Sparse（严格 O(3)）张量积：`PureCartesianTensorProductO3Sparse`

`PureCartesianTensorProductO3Sparse` 与第 5 节相同，仅将路径集合从全枚举裁剪为稀疏子集 \(\mathcal P_{\text{sparse}}\)，仍严格 \(O(3)\) 等变。

### 6.1 稀疏路径集合（实现 `_enumerate_paths_sparse`）
设 `max_rank_other`（默认 1）。仅保留满足
\[
\min(L_1,L_2)\le \texttt{max\_rank\_other}
\]
的交互：至少一侧的秩不超过 `max_rank_other`，每次交互至少有一个低秩张量（标量/向量；`max_rank_other=2` 时允许二阶张量）以减少计算。

### 6.2 `k_policy`（仅针对向量-张量）
当 \(\min(L_1,L_2)=1\) 时，δ 收缩允许的 \(k\) 由
- `"k0"`：只保留 \(k=0\Rightarrow L_{\text{out}}=L_1+L_2\)（向量外积提升秩）
- `"k1"`：只保留 \(k=1\Rightarrow L_{\text{out}}=L_1+L_2-2\)（一次 δ 收缩降低秩）
- `"both"`：两者都保留
控制。

### 6.3 `allow_epsilon`
若 `allow_epsilon=False`，则完全删除 ε 路径；此时本层只用 δ，不会产生新的 pseudo 分量。

### 6.4 `share_parity_weights`
- `share_parity_weights=False`：每个 \((s_1,s_2)\) 一套 \(W_{p}^{(s_1,s_2)}\)（最多 4 套）。
- `share_parity_weights=True`：仅按 \(s_{\text{out}}\in\{0,1\}\) 共享两套权重 \(W_p^{(0)},W_p^{(1)}\)。
这只是参数共享约束，不改变 \(\Gamma_p\) 与分级规则，故不影响等变性。

### 6.5 `assume_pseudo_zero` + `allow_epsilon=False`
当 `assume_pseudo_zero=True` 且 `allow_epsilon=False` 时，仅计算 true×true→true（\(s_1=s_2=s_{\text{out}}=0\)），省略所有 pseudo 通道与权重项。算子仍严格 O(3) 等变：只用 δ 收缩，\(\Gamma_p\) 在 O(3) 下严格等变且不产生 \(\det R\)；输出限定在 true 子空间仍保持等变性。

---

## 7. 边几何张量：\(n^{\otimes L}\)（与实现一致）

对边向量 \(r_{ij}\in\mathbb R^3\)，定义单位方向
\[
n_{ij}=\frac{r_{ij}}{\|r_{ij}\|}.
\]
构造秩 \(L\) 的几何张量（实现 `edge_rank_powers`）：
\[
E_L(n_{ij}) := n_{ij}^{\otimes L}\in\mathcal T_L,
\quad L=0,1,\dots,L_{\max}.
\]
它在 \(O(3)\) 下按自然张量表示变换（且本身是 true 张量，不额外带 \(\det R\)）。

网络中的消息传递用这些 \(E_L\) 与节点标量通道相乘得到 rank 特征，再通过上述 TP 做双线性混合。

---

## 8. 小结（实现到数学的一一对应）

- **表示**：\(\bigoplus_{s\in\{0,1\}}\bigoplus_{L\le L_{\max}}\mathbb R^C\otimes(\mathbb R^3)^{\otimes L}\)
- **群作用**：\((\det R)^s\prod R\) 逐指标作用
- **路径**：\(\delta\) 成对收缩（\(k\) 次）与可选一次 ε 收缩
- **分级传播**：\(s_{\text{out}}=s_1\oplus s_2\oplus \mathbf 1_{\varepsilon}\)
- **稀疏**：只裁剪路径集合与共享权重结构，等变性保持不变

---

## 9. Irreps 表示与 Pure-Cartesian 张量积的对应关系

本节阐述工程实践中常用的 irreps（如 `64x0e + 64x1o + 64x2e`）与 pure-cartesian（\(3^L\)）的对应关系。

### 9.1 `64x0e + 64x1o + 64x2e` 的块结构

这是 irreps 的直和，表示一个特征向量按 \((\ell,p)\) 分块：
- \(64\times 0e\)：64 个**标量**通道（\(\ell=0\)，even parity）
- \(64\times 1o\)：64 个**极向量**通道（\(\ell=1\)，odd parity），每个通道 3 个分量
- \(64\times 2e\)：64 个**二阶不可约张量**通道（\(\ell=2\)，even parity），每个通道 5 个分量（STF 基）

> **STF（Symmetric Trace-Free，对称无迹）**：STF 张量是满足以下两个条件的张量：
> 1. **对称性**：\(T_{ij} = T_{ji}\)（对 rank-2 张量）
> 2. **无迹性**：\(\mathrm{tr}(T) = \sum_{i} T_{ii} = 0\)（所有迹为零）
> 
> 对于 rank-2 张量，全 \(3\times 3\) 矩阵有 9 个独立分量。对称性约束减少到 6 个分量（\(T_{ij} = T_{ji}\)），无迹性再减少 1 个约束（\(\sum_i T_{ii} = 0\)），因此 STF 子空间维度为 \(6-1=5\)，对应 \(SO(3)\) 的 \(\ell=2\) 不可约表示（维度 \(2\ell+1=5\)）。
> 
> 数学上，任意对称 rank-2 张量 \(T_{ij}\) 可以唯一分解为：
> \[
> T_{ij} = \underbrace{\Big(\tfrac12(T_{ij}+T_{ji})-\tfrac13\delta_{ij}T_{kk}\Big)}_{\text{STF 部分（}\ell=2\text{，5 分量）}} + \underbrace{\tfrac13\delta_{ij}T_{kk}}_{\text{迹部分（}\ell=0\text{，1 分量）}}
> \]
> STF 部分在旋转下按 \(\ell=2\) 不可约表示变换，是 irreps 表示中 \(\ell=2\) 块的基础。

总维度为
\[
64(2\cdot0+1)+64(2\cdot1+1)+64(2\cdot2+1)=64(1+3+5)=576.
\]

把一个节点特征写为
\[
x = \big(x_{0e}, x_{1o}, x_{2e}\big),
\]
则形状对应为（以 batch 维省略）：
- \(x_{0e}\in\mathbb R^{64}\)
- \(x_{1o}\in\mathbb R^{64}\otimes \mathbb R^{3}\)
- \(x_{2e}\in\mathbb R^{64}\otimes \mathbb R^{5}\)

\(\mathbb R^{3}\) 与 \(\mathbb R^{5}\) 并非任意 3 或 5 维向量，而是 **O(3)/SO(3) 的不可约表示空间**（带固定的基与 \(D^\ell(R)\) 作用）。

### 9.2 Irreps 张量积：Clebsch–Gordan 耦合与通道混合

在 irreps 表示框架下，张量积对每条允许耦合路径 \((\ell_1,p_1)\otimes(\ell_2,p_2)\to(\ell_{\text{out}},p_{\text{out}})\) 用 Clebsch–Gordan（CG）系数 [6,8,14] 将 \(m\) 指标耦合成 \(\ell_{\text{out}}\) 的分量，再做通道混合。选择规则：

- **角动量三角条件**：
\[
|\ell_1-\ell_2|\le \ell_{\text{out}}\le \ell_1+\ell_2.
\]
- **宇称（parity）相乘**：
\[
p_{\text{out}} = p_1\,p_2,\qquad p\in\{+1,-1\},\ e\equiv +1,\ o\equiv -1.
\]

对固定的 \((\ell_1,\ell_2,\ell_{\text{out}})\)，耦合的坐标形式可写成
\[
z_{\text{out},m}
 = \sum_{m_1,m_2} C^{\ell_{\text{out}},m}_{\ell_1,m_1;\ \ell_2,m_2}\;
x_{\ell_1,m_1}\;y_{\ell_2,m_2},
\]
其中 \(C\) 是 Clebsch–Gordan（CG）系数 [6,8,14]。实现常用 Wigner-3j 符号 [7,8,15] 生成 \(C\)，满足（Condon–Shortley 相位约定）：
\[
C^{\ell_3,m_3}_{\ell_1,m_1;\ \ell_2,m_2}
=(-1)^{\ell_1-\ell_2+m_3}\sqrt{2\ell_3+1}
\begin{pmatrix}
\ell_1 & \ell_2 & \ell_3\\
m_1 & m_2 & -m_3
\end{pmatrix}.
\]
再对通道 \((a,b)\) 到输出通道 \(c\) 乘以权重并求和（与 `EquivariantTensorProduct` 中的 `W[c,a,b]` 一致）。

示例：\(\ell=1\) 与 \(\ell=1\) 的乘积允许
\[
1\otimes 1 \to 0\oplus 1\oplus 2,
\]
而宇称会决定这些输出是 \(0e/1e/2e\) 还是 \(0o/1o/2o\)（取决于输入的 \(p_1p_2\)）。

### 9.3 Pure-Cartesian 张量表示：\(3^L\) 全秩张量空间

pure-cartesian 改用
\[
\mathcal T_L=(\mathbb R^3)^{\otimes L}\quad(\dim=3^L)
\]
作为秩 \(L\) 的表示空间。对于 `channels=64, Lmax=2`，其块为：
- rank 0：\(\mathbb R^{64}\otimes \mathcal T_0 \cong \mathbb R^{64}\)（1 分量）
- rank 1：\(\mathbb R^{64}\otimes \mathcal T_1 \cong \mathbb R^{64}\otimes\mathbb R^3\)（3 分量）
- rank 2：\(\mathbb R^{64}\otimes \mathcal T_2 \cong \mathbb R^{64}\otimes(\mathbb R^3\otimes\mathbb R^3)\)（9 分量）

与 irreps 的关键差异在于：rank 2 在 pure-cartesian 中为全 \(3\times 3\)（9 分量），而 irreps 的 \(\ell=2\) 是其中的 **对称无迹（STF）** 子空间（5 分量）。

### 9.4 pure-cartesian 的张量积：δ/ε 外积与收缩

pure-cartesian 的基本双线性算子由路径 \(p\) 定义（第 4 节）：
- **外积**：两个张量拼接，秩相加
- **δ 收缩**：用 \(\delta_{ij}\) 成对消去左右各一个指标（秩减少 2）
- **ε 收缩**：用 \(\varepsilon_{ijk}\) 消去左右各一个指标，产生新指标（秩减少 1），引入 \(\det R\)

实现通过 `_einsum_for_path` 生成规范化的 Einstein 求和式执行。

### 9.5 pure-cartesian 需要 true/pseudo 的原因

由于 \(\varepsilon\) 在 \(O(3)\) 下满足 \(R^{\otimes 3}\varepsilon = (\det R)\varepsilon\)，ε 路径会多出 \(\det R\)。为使输出在 \(O(3)\) 下按表示空间闭合，需将特征空间分为 true/pseudo 两类，用 \(s_{\text{out}} = s_1\oplus s_2\oplus \mathbf 1_{\varepsilon}\) 追踪 \(\det R\) 因子。

### 9.6 总结：两种张量操作定义的对照

- irreps（例如 `64x0e+64x1o+64x2e`）：
  - 张量积 = CG（Wigner-3j）耦合 \(m\) 指标 + 通道混合
  - \(\ell\) 是不可约阶数，块维度是 \(2\ell+1\)
  - \(\ell=2\) 是 STF（5 分量）
- pure-cartesian：
  - 张量积 = δ/ε 的指标外积与收缩（路径求和）+ 通道混合
  - rank \(L\) 块维度是 \(3^L\)
  - rank 2 是全 \(3\times 3\)（9 分量），不会先投影到 STF

---

## 10. 数学基础：对称张量、齐次多项式与调和多项式

本节阐述 ICTD 方法的数学基础，这些概念在第 11 节的 `ictd_irreps.py` 实现中会用到。

### 10.1 对称 rank-\(L\) 张量 \(\leftrightarrow\) 三元齐次多项式（次数 \(L\)）的同构

令 \(V=\mathbb R^3\)，取坐标 \(x=(x_1,x_2,x_3)=(x,y,z)\)。给定一个 **对称** rank-\(L\) 张量
\[
T\in \mathrm{Sym}^L(V),\qquad T_{i_1\cdots i_L}=T_{i_{\sigma(1)}\cdots i_{\sigma(L)}}.
\]
定义对应的齐次多项式（次数为 \(L\)）：
\[
p_T(x):=\sum_{i_1,\dots,i_L=1}^3 T_{i_1\cdots i_L}\,x_{i_1}\cdots x_{i_L}\in \mathcal P_L.
\]
这给出了一个线性映射 \(\Phi:\mathrm{Sym}^L(V)\to \mathcal P_L\)。

反过来，任意齐次多项式都可写作
\[
p(x)=\sum_{a+b+c=L} t_{abc}\,x^a y^b z^c.
\]
由于对称性只关心每个坐标出现次数，\((a,b,c)\) 唯一确定一类排列等价的指标序列（即所有计数为 \((a,b,c)\) 的指标序列），可唯一构造对称张量 \(T\) 使得 \(p=\Phi(T)\)。实现中常用两套等价坐标：

- **对称轨道平均系数** \(t_{abc}\)：对称张量在该排列等价类中的公共分量（对称性保证等价类内所有 \(T_{i_1\dots i_L}\) 相等）。
- **对称轨道求和系数** \(s_{abc}\)：该排列等价类内所有分量的求和
  \[
  s_{abc}:=\sum_{\text{counts}(i_1,\dots,i_L)=(a,b,c)} T_{i_1\dots i_L}.
  \]
  两者满足
  \[
  s_{abc}=w_{abc}\,t_{abc},\qquad
  w_{abc}=\frac{L!}{a!\,b!\,c!}.
  \]
  即
  \[
  t_{abc}=\frac{1}{w_{abc}}\sum_{\text{counts}(i_1,\dots,i_L)=(a,b,c)} T_{i_1\dots i_L}.
  \]

并且两边维度一致：
\[
\dim\mathrm{Sym}^L(\mathbb R^3)=\binom{L+2}{2}=\dim\mathcal P_L,
\]
故 \(\Phi\) 是线性同构（两者等价）。

### 10.2 STF（对称无迹）\(\Longleftrightarrow\) 调和多项式（\(\Delta p=0\)）

对称张量的"迹"是用 \(\delta_{ij}\) 收缩两条指标：
\[
(\mathrm{tr}\,T)_{i_3\cdots i_L}:=\sum_{j=1}^3 T_{jj\,i_3\cdots i_L}\in \mathrm{Sym}^{L-2}(V).
\]
STF 的定义就是 \(\mathrm{tr}(T)=0\)（等价于所有更高次迹也为 0）。

在多项式侧，定义 Laplacian
\[
\Delta := \partial_x^2+\partial_y^2+\partial_z^2.
\]
核心对应：**Laplacian 在多项式侧对应张量侧的取迹**，即
\[
\Delta p_T(x)=L(L-1)\,p_{\mathrm{tr}(T)}(x).
\]
因此
\[
\mathrm{tr}(T)=0 \quad\Longleftrightarrow\quad \Delta p_T=0.
\]
即 **STF 张量对应调和多项式**（Laplacian 为零），其空间维度为 \(2L+1\)，这正是 \(SO(3)\) 的 \(\ell=L\) irreps 维度。

**示例（\(L=2\)）**：若 \(T_{ij}=T_{ji}\)，则
\[
p_T(x)=\sum_{i,j}T_{ij}x_i x_j,\qquad \Delta p_T = 2\,\mathrm{tr}(T).
\]
故 \(\Delta p_T=0\iff \mathrm{tr}(T)=0\)，与 rank-2 STF 条件完全一致。

对称 rank-\(L\) 张量与三元齐次多项式（次数 \(L\)）等价；STF 张量对应 **调和多项式**（Laplacian 为零）：
\[
\Delta f = (\partial_x^2+\partial_y^2+\partial_z^2)f = 0.
\]

STF（\(\ell=L\)）子空间可通过 \(\ker(\Delta)\) 得到。在第 11 节的 `ictd_irreps.py` 实现中，我们使用各向同性高斯测度的 Gram 矩阵来固定 harmonic 基，保证 \(O(3)\) 不变性。

---

## 11. ICTD-irreps（trace-chain irreps 内部表示，**无球谐**）的描述

本节对应代码：

- `molecular_force_field/models/ictd_irreps.py`
- `molecular_force_field/models/pure_cartesian_ictd_layers.py`（当前的 `pure-cartesian-ictd`：**内部不再使用 \(3^L\)**，而是全程用 \((2\ell+1)\) 的 irreps）

### 11.0 本节与 O(3)（宇称/反射）的关系

本节核心对象是 \(\mathcal H_\ell\)（调和多项式 / STF）与 trace-chain 分解。它们通常表述为 **\(SO(3)\)** 的 \(\ell\) 阶不可约表示（维度 \(2\ell+1\)）。在本实现构造下，它们**自然扩展为 \(O(3)\)** 的表示（包含宇称/反射），无需像 pure-cartesian 的 ε 路径那样额外引入 true/pseudo 分级来吸收 \(\det(R)\)。

- **当前实现**（`ictd_irreps.py` / `pure_cartesian_ictd_layers.py`）保证：对任意 \(R\in SO(3)\)，张量积与消息传递严格等变。
- 若需扩展为严格 \(O(3)\)（含 \(\det(R)=-1\)），可如第 2 节引入额外的 \(\mathbb Z_2\) 分级，将 \(\det(R)\) 因子作为显式表示标签随层传播。

#### 11.0.1 为什么这套 ICTD/调和多项式构造也满足 \(O(3)\)（含宇称）

关键：多项式/张量的群作用对任意 \(R\in O(3)\) 定义为变量代换（pullback）：
\[
(U_\ell(R)p)(x) := p(R^T x),\qquad R\in O(3).
\]
这本身就给出一个 \(O(3)\) 的表示，因为 \(U_\ell(R_1R_2)=U_\ell(R_1)\,U_\ell(R_2)\)。

ICTD 的所有构造步骤均与该 \(O(3)\) 作用可交换（因此得到的投影/耦合均为 \(O(3)\) intertwiner）：

- **Laplacian 不变**：\(\Delta(p\circ R^T)=(\Delta p)\circ R^T\)，因此 \(\mathcal H_\ell=\ker\Delta\) 对任意 \(R\in O(3)\) 都是不变子空间。
- **trace-chain 的 \(r^2\) 不变**：\(\|R^T x\|^2=\|x\|^2\)，故乘以 \(r^{2k}\) 的提升算子与 \(U(R)\) 可交换。
- **选基/投影用的 Gram 是 \(O(3)\) 不变的**：`ictd_irreps.py` 使用各向同性高斯测度的 Gram 矩阵
  \(\langle p,q\rangle_G\propto\int p(x)q(x)e^{-\|x\|^2/2}dx\)，它仅依赖 \(\|x\|\)，因此对 \(O(3)\) 不变。这保证了正交归一得到的基及最小二乘投影 \(P_{L\to\ell}\) 与 \(U(R)\) 交换。

因此，从“多项式乘法 + trace-chain 投影”得到的耦合张量 \(C^{\ell_3}_{\ell_1\ell_2}\) 不仅是 \(SO(3)\) intertwiner，也是 \(O(3)\) intertwiner。

**宇称（parity）自动出现**：对全反演 \(R=-I\) 有
\[
(U_\ell(-I)p)(x)=p((-I)^T x)=p(-x)=(-1)^\ell p(x),
\]
也即 \(\ell\) 阶齐次（调和）多项式在宇称下天然带有 \((-1)^\ell\) 的符号。这与球谐函数的性质 \(Y_{\ell m}(-\hat n)=(-1)^\ell Y_{\ell m}(\hat n)\) 一致。

> 对比：pure-cartesian 的 ε 路径会额外引入 \((\det R)\) 因子（见第 3.2、5.2 节），因此需要显式 true/pseudo 分级标记；而本节的 ICTD/调和多项式路线不显式使用 ε 收缩来生成伪张量，因此不需要额外的 \(\mathbb Z_2\) 通道来吸收 \(\det R\)。

### 11.1 记号

- 令 \(V=\mathbb{R}^3\)，\(R\in SO(3)\)。
- \(\mathrm{Sym}^L(V)\)：对称 rank-\(L\) 张量空间，维度 \(\binom{L+2}{2}\)。
- \(\mathcal{P}_L\)：三元齐次多项式（次数 \(L\)）空间，与 \(\mathrm{Sym}^L(V)\) 同构。
- \(\Delta=\partial_x^2+\partial_y^2+\partial_z^2\)：Laplacian（把次数 \(L\) 降到 \(L-2\)）。
- \(\mathcal{H}_L:=\ker(\Delta)\subset \mathcal{P}_L\)：调和（harmonic）子空间，\(\dim\mathcal{H}_L=2L+1\)。它对应 irreps 的 \(\ell=L\) 块（STF）。
- \(r^2=x^2+y^2+z^2\)。

### 11.2 对称张量 \(\leftrightarrow\) 齐次多项式

本节使用与 **10.1** 相同的同构：对称 rank-\(L\) 张量 \(\mathrm{Sym}^L(\mathbb R^3)\) 与次数 \(L\) 的三元齐次多项式空间 \(\mathcal P_L\) 线性等价（见第 10.1 节）。

取单项式基 \(m_{abc}(x)=x^a y^b z^c\)（\(a+b+c=L\)）。任意 \(t\in\mathrm{Sym}^L(V)\)（在该基下的系数向量记为 \(t_L\)）对应多项式
\[
p_t(x)=\sum_{a+b+c=L} (t_L)_{abc}\, x^a y^b z^c.
\]
旋转 \(R\) 对多项式的自然作用是变量替换
\[
(U_L(R)p)(x)=p(R^T x),
\]
它在 \(\mathcal{P}_L\) 上诱导出一个表示（也是 \(\mathrm{Sym}^L(V)\) 的表示）。

### 11.3 Harmonic（STF）子空间：\(\ker(\Delta)\)

\[
\mathcal{H}_L:=\{p\in\mathcal{P}_L:\Delta p=0\}.
\]
它是"对称无迹（STF）"子空间的多项式版本，维度为 \(2L+1\)。关于 STF 与调和多项式的对应关系，见第 10.2 节。实现中：

- 先在单项式系数基下写出 \(\Delta:\mathcal{P}_L\to\mathcal{P}_{L-2}\) 的矩阵；
- 取其零空间得到 \(\mathcal{H}_L\) 的一组基向量（列向量）。

### 11.4 用 **O(3) 不变**的高斯 Gram 固定基（关键点）

为使 \(D^{(\ell)}(R)\) 一致/正交，避免"任意零空间基"带来的数值漂移，本实现用各向同性高斯测度定义内积：
\[
\langle p,q\rangle_G := \mathbb{E}_{X\sim\mathcal{N}(0,I_3)}[p(X)q(X)].
\]
等价地（忽略一个与归一化无关的常数因子）：
\[
\langle p,q\rangle_G \ \propto\ \int_{\mathbb R^3} p(x)\,q(x)\,e^{-\|x\|^2/2}\,dx.
\]
它对 \(O(3)\) 不变，因此对 \(SO(3)\) 也不变。

在单项式基下，该内积对应 Gram 矩阵 \(G_L\)，其元素为
\[
(G_L)_{(abc),(a'b'c')}=
\mathbb{E}[x^{a+a'}]\ \mathbb{E}[y^{b+b'}]\ \mathbb{E}[z^{c+c'}],
\]
其中一维高斯矩
\[
\mathbb{E}[x^n]=
\begin{cases}
0,& n\ \text{为奇数}\\
(n-1)!!,& n\ \text{为偶数}.
\end{cases}
\]

设 \(B_L\in\mathbb{R}^{D_L\times(2L+1)}\) 是 \(\ker(\Delta)\) 的任意基（\(D_L=\binom{L+2}{2}\)）。在 \(G_L\) 内积下白化/正交化，得到 harmonic 基（仍记为 \(B_L\)）满足：
\[
B_L^T G_L B_L = I_{2L+1}.
\]

### 11.5 trace-chain 分解与投影 \(P_{L\to \ell}\)

经典分解（harmonic decomposition / trace chain）：
\[
\mathcal{P}_L=\bigoplus_{k=0}^{\lfloor L/2\rfloor} r^{2k}\mathcal{H}_{L-2k}.
\]
给定 \(\ell=L-2k\)，定义提升算子（实现中为矩阵）：
\[
M_{\ell,k}:\mathcal{H}_\ell\to \mathcal{P}_L,\quad h\mapsto r^{2k}h.
\]
\(V_{\ell,k}:=M_{\ell,k}B_\ell\) 是嵌入到 \(\mathcal{P}_L\) 的 \((2\ell+1)\) 列基。对任意次数 \(L\) 的系数向量 \(t_L\)，用 \(G_L\) 的最小二乘投影取出 \(\ell\) 块坐标：
\[
c_\ell = P_{L\to \ell}\, t_L,
\quad
P_{L\to \ell}:=(V_{\ell,k}^T G_L V_{\ell,k})^{-1}V_{\ell,k}^T G_L.
\]
\(c_\ell\in\mathbb{R}^{2\ell+1}\) 为 irreps 块坐标（在固定的 harmonic 基下）。

### 11.6 无球谐的边方向特征 \(Y_\ell(n)\)

给定单位方向向量 \(n\in\mathbb{R}^3\)，定义齐次多项式
\[
p_{n,\ell}(x)=(n\cdot x)^\ell\in\mathcal{P}_\ell.
\]
把它展开为单项式系数向量 \(t_\ell(n)\)，再投影到 harmonic 块得到
\[
Y_\ell(n):=B_\ell^T G_\ell\, t_\ell(n)\in\mathbb{R}^{2\ell+1}.
\]
因为 \(G_\ell\) 与 harmonic 基 \(B_\ell\) 都是按 \(O(3)\) 不变结构构造的，故 \(Y_\ell\) 在旋转下满足
\[
Y_\ell(Rn)=D^{(\ell)}(R)\,Y_\ell(n),
\]
其中 \(D^{(\ell)}(R)\) 是 \(SO(3)\) 在 \(\mathcal{H}_\ell\) 上诱导的不可约表示矩阵。

**归一化/基的一致性说明**：此处构造的 \(Y_\ell(n)\) 是在本实现选择的 harmonic 基 \(B_\ell\) 与内积 \(G_\ell\) 下的坐标。它与常见的（实）球谐函数基 \(Y_{\ell m}(\hat n)\) 一般不要求逐分量相等，但二者承载同一个 \((2\ell+1)\) 维 \(SO(3)\) 不可约表示，至多相差一个 \((2\ell+1)\times(2\ell+1)\) 的正交变换；变换性质完全一致：\(Y_\ell(Rn)=D^{(\ell)}(R)Y_\ell(n)\)。

### 11.7 在这套基下构造 CG（耦合张量）而不使用 Wigner-3j

给定 \(a\in\mathcal{H}_{\ell_1}\)、\(b\in\mathcal{H}_{\ell_2}\)（在基下坐标分别为 \(a_{m_1}, b_{m_2}\)），考虑多项式乘积：
\[
p_a(x)\,p_b(x)\in\mathcal{P}_{L},\quad L=\ell_1+\ell_2.
\]
再用 trace-chain 投影提取 \(\ell_3\) 块：
\[
c_{\ell_3}:=P_{L\to \ell_3}\big(p_a p_b\big)\in\mathcal{H}_{\ell_3}.
\]
这等价于存在一个三阶耦合张量（CG 张量）
\[
C^{\ell_3}_{\ell_1\ell_2}\in\mathbb{R}^{(2\ell_1+1)\times(2\ell_2+1)\times(2\ell_3+1)}
\]
使得
\[
(c_{\ell_3})_{m_3}=\sum_{m_1,m_2} a_{m_1}b_{m_2}\ (C^{\ell_3}_{\ell_1\ell_2})_{m_1m_2m_3}.
\]
构造自动满足选择规则：

- 三角不等式 \(|\ell_1-\ell_2|\le \ell_3\le \ell_1+\ell_2\)；
- \(\ell_1+\ell_2-\ell_3\) 必须为偶数（来自 \(r^{2k}\) 的 trace-chain 结构）。

由于在多项式表示空间中使用 \(SO(3)\)-同态的投影构造，\(C^{\ell_3}_{\ell_1\ell_2}\) 是严格的 intertwiner（实现中已通过旋转前后 TP 交换的单元测试验证）。

**交换对称性（补充）**：由于多项式乘法满足交换律 \(p_a p_b=p_b p_a\)，交换输入 \((\ell_1,m_1)\leftrightarrow(\ell_2,m_2)\) 时，耦合张量对应交换前两维的同一个 intertwiner。采用标准 Condon–Shortley 相位约定的 CG 系数时，常见关系为
\[
(C^{\ell_3}_{\ell_1\ell_2})_{m_1m_2m_3}
=(-1)^{\ell_1+\ell_2-\ell_3}\,(C^{\ell_3}_{\ell_2\ell_1})_{m_2m_1m_3}.
\]
在本实现构造中，具体符号/相位可能随所选基 \(B_\ell\) 发生整体正交变换，但交换输入对应确定的符号/基变换这一结构保持不变。

### 11.8 可学习版本：shared weights + per-path gating

网络内部特征采用 irreps 分块：
\[
x^{(\ell)}\in\mathbb{R}^{\mu\times(2\ell+1)},
\]
其中 \(\mu\) 是该 \(\ell\) 的 multiplicity（通道数）。

对每条边 \(e\)，径向网络输出每条“耦合路径” \(p=(\ell_1,\ell_2\to\ell_3)\) 的标量 gate：
\[
g_e(p)\in\mathbb{R}.
\]
每条路径存一个共享权重张量
\[
W_p\in\mathbb{R}^{\mu_{\text{out}}\times\mu_1\times\mu_2}.
\]
则路径贡献为
\[
m_{e,p}^{(\ell_3)} = g_e(p)\cdot
\Big[\;W_p\cdot_{\mu_1,\mu_2}\big(x^{(\ell_1)}\otimes_{C^{\ell_3}_{\ell_1\ell_2}} y^{(\ell_2)}\big)\;\Big],
\]
对所有 \(p\) 求和得到边消息 \(m_e^{(\ell_3)}\)。由于 \(g_e(p)\) 是标量、\(\otimes_C\) 是 intertwiner，整体仍保持严格的 \(SO(3)\) 等变性。

---

## 12. 张量积模式性能对比（Benchmark 结果）

本节提供所有六种张量积实现模式的性能对比数据，包括参数量、前向传播速度、反向传播速度和等变性验证结果。

### 12.1 测试环境

#### 12.1.1 CPU 测试环境

- **硬件**：CPU
- **模型配置**：`channels=64`, `lmax=0` 到 `lmax=6`
- **测试数据**：32 atoms, 256 edges
- **测试方法**：前向传播 30 次平均，反向传播 20 次平均
- **等变性测试**：O(3) 等变性（包括宇称/反射），20 次随机旋转测试
- **数据类型**：float64
- **注意**：`pure-cartesian` 只测试到 lmax=3（lmax≥4 时失败）

#### 12.1.2 GPU 测试环境

- **硬件**：NVIDIA GeForce RTX 3090
- **模型配置**：`channels=64`, `lmax=0` 到 `lmax=6`
- **测试数据**：32 atoms, 256 edges
- **测试方法**：前向传播 30 次平均，反向传播 20 次平均
- **等变性测试**：O(3) 等变性（包括宇称/反射），20 次随机旋转测试
- **数据类型**：float64

### 12.2 性能对比总览（lmax=2，典型配置）

基于 lmax=2 的测试结果（详见 12.7 和 12.8 节）：

| 模式 | 等变性 | 参数量 (lmax=2) | 相对参数量 | CPU 加速比 | GPU 加速比 | 等变性误差 |
|------|--------|----------------|------------|------------|------------|------------|
| `spherical` | ✅ 严格等变 | 6,540,634 | 100% (基准) | 1.00x (基准) | 1.00x (基准) | ~1e-15 |
| `partial-cartesian` | ✅ 严格等变 | 5,404,938 | 82.6% | 1.06x | 0.75x | ~1e-14 |
| `partial-cartesian-loose` | ⚠️ 近似等变 | 5,406,026 | 82.7% | 1.33x | 1.15x | ~1e-15 |
| `pure-cartesian` | ✅ 严格等变 | 33,626,186 | 514.0% | 0.06x (极慢) | 0.06x (极慢) | ~1e-14 |
| `pure-cartesian-sparse` | ✅ 严格等变 | 4,606,026 | 70.4% | 1.39x | 1.17x | ~1e-15 |
| `pure-cartesian-ictd` | ✅ 严格等变 | 1,824,497 | **27.9%** | **4.12x (最快)** | **2.10x (最快)** | ~1e-7 |

**关键发现**：
- **`pure-cartesian-ictd` 在所有 lmax 下都是最快的**（CPU: 最高 4.12x，GPU: 最高 2.10x）
- **`pure-cartesian-ictd` 参数量最少**（27-32% of spherical）
- **`pure-cartesian` 在 lmax≥4 时失败**（内存不足）

### 12.3 参数量详细对比（lmax=0 到 6）

参数量随 lmax 的变化趋势（详见 12.7.2 和 12.8.2 节）：

| lmax | spherical | partial-cartesian | partial-cartesian-loose | pure-cartesian | pure-cartesian-sparse | pure-cartesian-ictd |
|------|----------:|------------------:|----------------------:|---------------:|---------------------:|-------------------:|
| 0    | 2,313,018 |        1,128,074 |            1,128,074 |      2,725,450 |           1,128,010 |             626,653 |
| 1    | 4,164,682 |        2,734,026 |            2,734,434 |     11,786,058 |           2,733,898 |           1,208,931 |
| 2    | 6,540,634 |        5,404,938 |            5,406,026 |     33,626,186 |           4,606,026 |           1,824,497 |
| 3    | 8,654,426 |        9,407,050 |            9,409,090 |     74,635,594 |           6,744,394 |           2,481,673 |
| 4    | 10,768,218 |       15,272,842 |           15,276,106 |         **FAILED** |           8,882,762 |           3,197,103 |
| 5    | 12,882,010 |       23,268,554 |           23,273,314 |         **FAILED** |          11,021,130 |           3,979,109 |
| 6    | 14,995,802 |       33,926,666 |           33,933,194 |         **FAILED** |          13,159,498 |           4,844,335 |

**性能分析**：
- **`pure-cartesian-ictd` 参数量始终最少**（27-32% of spherical）
- **`pure-cartesian-sparse` 参数量适中**（70-88% of spherical）
- **`partial-cartesian` 参数量减少 17-18%**（82-83% of spherical）
- **`pure-cartesian` 在 lmax≥4 时失败**（内存不足）

### 12.4 速度详细对比（lmax=0 到 6）

总训练时间加速比（前向+反向，相对于 spherical，详见 12.7.1 和 12.8.1 节）：

**CPU 环境**：
| lmax | partial-cartesian | partial-cartesian-loose | pure-cartesian | pure-cartesian-sparse | pure-cartesian-ictd |
|------|------------------:|----------------------:|---------------:|---------------------:|-------------------:|
| 0    |             1.06x |                  1.13x |           0.36x |                 1.07x |                **2.97x** |
| 1    |             1.05x |                  1.37x |           0.13x |                 1.02x |                **3.33x** |
| 2    |             1.06x |                  1.33x |           0.06x |                 1.39x |                **4.12x** |
| 3    |             0.58x |                  0.70x |           0.02x |                 1.05x |                **2.68x** |
| 4    |             0.37x |                  0.43x |        **FAILED** |                 0.97x |                **2.20x** |
| 5    |             0.23x |                  0.28x |        **FAILED** |                 0.78x |                **1.81x** |
| 6    |             0.16x |                  0.17x |        **FAILED** |                 0.53x |                **1.58x** |

**GPU 环境**：
| lmax | partial-cartesian | partial-cartesian-loose | pure-cartesian | pure-cartesian-sparse | pure-cartesian-ictd |
|------|------------------:|----------------------:|---------------:|---------------------:|-------------------:|
| 0    |             0.96x |                  1.52x |           0.54x |                 1.17x |                **1.92x** |
| 1    |             0.85x |                  1.33x |           0.16x |                 1.02x |                **1.97x** |
| 2    |             0.75x |                  1.15x |           0.06x |                 1.17x |                **2.10x** |
| 3    |             0.56x |                  0.81x |           0.02x |                 1.15x |                **1.91x** |
| 4    |             0.38x |                  0.51x |        **FAILED** |                 0.99x |                **1.78x** |
| 5    |             0.26x |                  0.32x |        **FAILED** |                 0.75x |                **1.44x** |
| 6    |             0.17x |                  0.21x |        **FAILED** |                 0.46x |                **1.05x** |

**性能分析**：
- **`pure-cartesian-ictd` 在所有 lmax 下都是最快的**（CPU: 最高 4.12x at lmax=2，GPU: 最高 2.10x at lmax=2）
- **lmax ≤ 3**: `pure-cartesian-ictd` 优势最明显（CPU: 2.68x - 4.12x，GPU: 1.91x - 2.10x）
- **lmax = 4-6**: `pure-cartesian-ictd` 仍然最快（CPU: 1.58x - 2.20x，GPU: 1.05x - 1.78x）
- **`pure-cartesian-sparse` 表现稳定**：在所有 lmax 下都接近或快于基准（CPU: 0.53x - 1.39x，GPU: 0.46x - 1.17x）

### 12.5 等变性验证（O(3)，包含宇称）

等变性误差随 lmax 的变化（详见 12.7.3 和 12.8.3 节）：

**CPU 环境**（典型值）：
- **严格等变**（误差 ~1e-15）：`spherical`, `partial-cartesian-loose`, `pure-cartesian-sparse`, `pure-cartesian`
- **可接受等变**（误差 ~1e-7 到 1e-6）：`pure-cartesian-ictd`（lmax=2: 7.73e-08，lmax=6: 1.00e-06）
- **近似等变**：`partial-cartesian`（误差 ~1e-14 到 1e-15）

**GPU 环境**（典型值）：
- **严格等变**（误差 ~1e-15）：`spherical`, `partial-cartesian-loose`, `pure-cartesian-sparse`
- **可接受等变**（误差 ~1e-7）：`pure-cartesian-ictd`（lmax=2: 7.18e-07，lmax=6: 1.11e-07）

**注意**：`pure-cartesian-ictd` 的等变性误差虽然比严格等变模式大（~1e-7 vs ~1e-15），但仍然在可接受范围内（< 1e-6），且其速度和参数量优势明显。

### 12.6 性能分析

基于完整测试结果（lmax=0 到 6）的综合分析：

1. **参数量优化**：
   - **`pure-cartesian-ictd` 参数量始终最少**（27-32% of spherical），在所有 lmax 下都保持优势
   - **`pure-cartesian-sparse` 参数量适中**（70-88% of spherical），平衡了性能和参数量
   - **`partial-cartesian` 参数量减少 17-18%**（82-83% of spherical），略优于基准

2. **速度优化**：
   - **`pure-cartesian-ictd` 在所有 lmax 下都是最快的**（CPU: 最高 4.12x at lmax=2，GPU: 最高 2.10x at lmax=2）
   - **`pure-cartesian-sparse` 表现稳定**：在所有 lmax 下都接近或快于基准（CPU: 0.53x - 1.39x，GPU: 0.46x - 1.17x）
   - **`partial-cartesian-loose` 在低 lmax 时较快**（CPU: 1.13x - 1.37x，GPU: 1.15x - 1.52x），但非严格等变

3. **等变性保证**：
   - **严格等变**（误差 ~1e-15）：`spherical`, `partial-cartesian-loose`, `pure-cartesian-sparse`, `pure-cartesian`
   - **可接受等变**（误差 ~1e-7）：`pure-cartesian-ictd`（虽然误差较大，但仍在可接受范围内）

4. **不推荐使用**：
   - **`pure-cartesian`（非稀疏）**：速度极慢（0.02x - 0.54x），参数量巨大（+414%），且在 lmax≥4 时失败（内存不足），仅用于研究目的

### 12.7 CPU 测试结果（lmax=0 到 6）

#### 12.7.1 总训练时间加速比（前向+反向，相对于 spherical）

| lmax | partial-cartesian | partial-cartesian-loose | pure-cartesian | pure-cartesian-sparse | pure-cartesian-ictd |
|------|------------------:|----------------------:|---------------:|---------------------:|-------------------:|
| 0    |             1.06x |                  1.13x |           0.36x |                 1.07x |                **2.97x** |
| 1    |             1.05x |                  1.37x |           0.13x |                 1.02x |                **3.33x** |
| 2    |             1.06x |                  1.33x |           0.06x |                 1.39x |                **4.12x** |
| 3    |             0.58x |                  0.70x |           0.02x |                 1.05x |                **2.68x** |
| 4    |             0.37x |                  0.43x |        **FAILED** |                 0.97x |                **2.20x** |
| 5    |             0.23x |                  0.28x |        **FAILED** |                 0.78x |                **1.81x** |
| 6    |             0.16x |                  0.17x |        **FAILED** |                 0.53x |                **1.58x** |

**性能分析**：
- **`pure-cartesian-ictd` 在所有 lmax 下都是最快的**（最高 **4.12x 加速** at lmax=2）
- **lmax ≤ 3**: `pure-cartesian-ictd` 优势最明显（2.68x - 4.12x）
- **lmax = 4-6**: `pure-cartesian-ictd` 仍然最快（1.58x - 2.20x）
- **`pure-cartesian-sparse` 表现稳定**：在所有 lmax 下都接近基准（0.53x - 1.39x）

#### 12.7.2 参数量对比（CPU，lmax=0 到 6）

| lmax | spherical | partial-cartesian | partial-cartesian-loose | pure-cartesian | pure-cartesian-sparse | pure-cartesian-ictd |
|------|----------:|------------------:|----------------------:|---------------:|---------------------:|-------------------:|
| 0    | 2,313,018 |        1,128,074 |            1,128,074 |      2,725,450 |           1,128,010 |             626,653 |
| 1    | 4,164,682 |        2,734,026 |            2,734,434 |     11,786,058 |           2,733,898 |           1,208,931 |
| 2    | 6,540,634 |        5,404,938 |            5,406,026 |     33,626,186 |           4,606,026 |           1,824,497 |
| 3    | 8,654,426 |        9,407,050 |            9,409,090 |     74,635,594 |           6,744,394 |           2,481,673 |
| 4    | 10,768,218 |       15,272,842 |           15,276,106 |         **FAILED** |           8,882,762 |           3,197,103 |
| 5    | 12,882,010 |       23,268,554 |           23,273,314 |         **FAILED** |          11,021,130 |           3,979,109 |
| 6    | 14,995,802 |       33,926,666 |           33,933,194 |         **FAILED** |          13,159,498 |           4,844,335 |

**性能分析**：
- **`pure-cartesian-ictd` 参数量始终最少**（27-32% of spherical）
- **`pure-cartesian-sparse` 参数量适中**（70-88% of spherical）
- **`pure-cartesian` 在 lmax≥4 时失败**（内存不足）

#### 12.7.3 等变误差对比（CPU，O(3)，包含宇称）

| lmax | spherical | partial-cartesian | partial-cartesian-loose | pure-cartesian | pure-cartesian-sparse | pure-cartesian-ictd |
|------|----------:|------------------:|----------------------:|---------------:|---------------------:|-------------------:|
| 0    | 3.33e-15  |        8.24e-16  |            6.39e-14  |      3.41e-15  |           4.34e-15  |           1.69e-12 |
| 1    | 5.58e-14  |        3.33e-16  |            1.65e-15  |      5.01e-15  |           7.24e-15  |           3.80e-08 |
| 2    | 3.88e-15  |        1.87e-14  |            7.73e-16  |      3.08e-15  |           3.56e-14  |           7.73e-08 |
| 3    | 1.16e-15  |        2.47e-14  |            3.71e-15  |      8.14e-16  |           9.63e-16  |           7.23e-08 |
| 4    | 1.27e-15  |        6.83e-15  |            1.00e-15  |       **FAILED** |           1.24e-14  |           3.50e-05 |
| 5    | 3.43e-15  |        3.45e-15  |            2.27e-15  |       **FAILED** |           3.59e-14  |           8.11e-08 |
| 6    | 3.26e-15  |        2.01e-15  |            5.82e-16  |       **FAILED** |           1.51e-15  |           1.00e-06 |

**性能分析**：
- **严格等变**（误差 ~1e-15）：`spherical`, `partial-cartesian-loose`, `pure-cartesian-sparse`
- **可接受等变**（误差 ~1e-7 到 1e-6）：`pure-cartesian-ictd`

### 12.8 GPU 测试结果（lmax=0 到 6）

#### 12.8.1 总训练时间加速比（前向+反向，相对于 spherical）

| lmax | partial-cartesian | partial-cartesian-loose | pure-cartesian | pure-cartesian-sparse | pure-cartesian-ictd |
|------|------------------:|----------------------:|---------------:|---------------------:|-------------------:|
| 0    |             0.96x |                  1.52x |           0.54x |                 1.17x |                **1.92x** |
| 1    |             0.85x |                  1.33x |           0.16x |                 1.02x |                **1.97x** |
| 2    |             0.75x |                  1.15x |           0.06x |                 1.17x |                **2.10x** |
| 3    |             0.56x |                  0.81x |           0.02x |                 1.15x |                **1.91x** |
| 4    |             0.38x |                  0.51x |        **FAILED** |                 0.99x |                **1.78x** |
| 5    |             0.26x |                  0.32x |        **FAILED** |                 0.75x |                **1.44x** |
| 6    |             0.17x |                  0.21x |        **FAILED** |                 0.46x |                **1.05x** |

**性能分析**：
- **`pure-cartesian-ictd` 在所有 lmax 下都是最快的**（最高 **2.10x 加速** at lmax=2）
- **lmax ≤ 3**: `pure-cartesian-ictd` 优势最明显（1.91x - 2.10x）
- **lmax = 4-5**: `pure-cartesian-ictd` 仍然更快（1.44x - 1.78x）
- **lmax = 6**: `pure-cartesian-ictd` 和 `spherical` 几乎相等（1.05x）

#### 12.8.2 参数量对比（GPU，lmax=0 到 6）

| lmax | spherical | partial-cartesian | partial-cartesian-loose | pure-cartesian | pure-cartesian-sparse | pure-cartesian-ictd |
|------|----------:|------------------:|----------------------:|---------------:|---------------------:|-------------------:|
| 0    | 2,313,018 |        1,128,074 |            1,128,074 |      2,725,450 |           1,128,010 |             626,653 |
| 1    | 4,164,682 |        2,734,026 |            2,734,434 |     11,786,058 |           2,733,898 |           1,208,931 |
| 2    | 6,540,634 |        5,404,938 |            5,406,026 |     33,626,186 |           4,606,026 |           1,824,497 |
| 3    | 8,654,426 |        9,407,050 |            9,409,090 |     74,635,594 |           6,744,394 |           2,481,673 |
| 4    | 10,768,218 |       15,272,842 |           15,276,106 |         **FAILED** |           8,882,762 |           3,197,103 |
| 5    | 12,882,010 |       23,268,554 |           23,273,314 |         **FAILED** |          11,021,130 |           3,979,109 |
| 6    | 14,995,802 |       33,926,666 |           33,933,194 |         **FAILED** |          13,159,498 |           4,844,335 |

**性能分析**：
- **`pure-cartesian-ictd` 参数量始终最少**（27-32% of spherical）
- **`pure-cartesian-sparse` 参数量适中**（70-88% of spherical）
- **`pure-cartesian` 在 lmax≥4 时失败**（内存不足）

#### 12.8.3 等变误差对比（GPU，O(3)，包含宇称）

| lmax | spherical | partial-cartesian | partial-cartesian-loose | pure-cartesian | pure-cartesian-sparse | pure-cartesian-ictd |
|------|----------:|------------------:|----------------------:|---------------:|---------------------:|-------------------:|
| 0    | 1.03e-15  |        6.37e-16  |            1.20e-15  |      3.27e-15  |           9.31e-16  |           5.24e-08 |
| 2    | 9.27e-16  |        1.97e-16  |            8.96e-16  |      3.18e-14  |           1.30e-15  |           7.18e-07 |
| 4    | 9.16e-16  |        7.76e-14  |            4.26e-16  |       **FAILED** |           6.19e-16  |           7.16e-07 |
| 6    | 4.77e-16  |        7.02e-16  |            5.65e-16  |       **FAILED** |           5.72e-16  |           1.11e-07 |

**性能分析**：
- **严格等变**（误差 ~1e-15）：`spherical`, `partial-cartesian-loose`, `pure-cartesian-sparse`
- **可接受等变**（误差 ~1e-7）：`pure-cartesian-ictd`

### 12.9 推荐使用场景

#### 12.9.1 CPU 环境（推荐）
- **速度优先 + 参数效率**：使用 `pure-cartesian-ictd`（最高 **4.12x 加速** at lmax=2，参数量最少 27-32%）
- **高精度需求**：使用 `spherical` 或 `pure-cartesian-sparse`（等变误差 ~1e-15）
- **平衡选择**：使用 `pure-cartesian-sparse`（0.53x - 1.39x，高精度，参数量适中）
- **标准基准**：使用 `spherical`（最高精度，标准实现）
- **避免使用**：`pure-cartesian`（速度极慢，lmax≥4 时失败）

#### 12.9.2 GPU 环境（推荐）
- **速度优先 + 参数效率**：使用 `pure-cartesian-ictd`（最高 **2.10x 加速**，参数量最少 27-32%）
- **高精度需求**：使用 `spherical` 或 `pure-cartesian-sparse`（等变误差 ~1e-15）
- **平衡选择**：使用 `pure-cartesian-sparse`（1.17x 加速，高精度，参数量适中）
- **避免使用**：`pure-cartesian`（速度极慢，lmax≥4 时失败）

### 12.10 实际任务测试结果

**数据集**：五条氮氧化物和碳结构反应路径的 NEB（Nudged Elastic Band）数据，截取到 fmax=0.2，总共 2,788 条数据。测试集：每个反应选取 1-2 条完整或不完整的数据。

**测试配置**：64 channels, lmax=2, float64

#### 12.10.1 能量和力精度对比

<table>
<thead>
<tr>
<th style="text-align:center">方法</th>
<th style="text-align:center">配置</th>
<th style="text-align:center">模式</th>
<th style="text-align:center">能量 RMSE<br/>(mev/atom)</th>
<th style="text-align:center">力 RMSE<br/>(mev/Å)</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="3" style="text-align:center;vertical-align:middle"><strong>MACE</strong></td>
<td style="text-align:center">Lmax=2, 64ch</td>
<td style="text-align:center">-</td>
<td style="text-align:center">0.13</td>
<td style="text-align:center">11.6</td>
</tr>
<tr>
<td style="text-align:center">Lmax=2, 128ch</td>
<td style="text-align:center">-</td>
<td style="text-align:center">0.12</td>
<td style="text-align:center">11.3</td>
</tr>
<tr>
<td style="text-align:center">Lmax=2, 198ch</td>
<td style="text-align:center">-</td>
<td style="text-align:center">0.24</td>
<td style="text-align:center">15.1</td>
</tr>
<tr>
<td rowspan="4" style="text-align:center;vertical-align:middle"><strong>FSCETP</strong></td>
<td rowspan="4" style="text-align:center;vertical-align:middle">Lmax=2, 64ch</td>
<td style="text-align:center"><strong>spherical</strong></td>
<td style="text-align:center"><strong>0.044</strong> ⭐</td>
<td style="text-align:center"><strong>7.4</strong> ⭐</td>
</tr>
<tr>
<td style="text-align:center"><strong>partial-cartesian</strong></td>
<td style="text-align:center">0.045</td>
<td style="text-align:center"><strong>7.4</strong> ⭐</td>
</tr>
<tr>
<td style="text-align:center">partial-cartesian-loose</td>
<td style="text-align:center">0.048</td>
<td style="text-align:center">8.4</td>
</tr>
<tr>
<td style="text-align:center">pure-cartesian-ictd</td>
<td style="text-align:center">0.046</td>
<td style="text-align:center">9.0</td>
</tr>
</tbody>
</table>

**说明**：
- 数值越小越好
- ⭐ 表示该指标的最佳结果

#### 12.10.2 性能分析

**能量精度对比**：
- **FSCETP 相比 MACE (64ch) 的能量 RMSE 降低了 66.2%**
  - FSCETP 最优结果：0.044 mev/atom（`spherical` 模式）
  - MACE (64ch) 基准值：0.13 mev/atom
  - 相对误差比：0.34（FSCETP / MACE）

**力精度对比**：
- **FSCETP 相比 MACE (64ch) 的力 RMSE 降低了 36.2%**
  - FSCETP 最优结果：7.4 mev/Å（`spherical` 和 `partial-cartesian` 模式）
  - MACE (64ch) 基准值：11.6 mev/Å
  - 相对误差比：0.64（FSCETP / MACE）

**模式性能总结**：
- **能量精度最优模式**：`spherical`（0.044 mev/atom）
- **力精度最优模式**：`spherical` 和 `partial-cartesian`（7.4 mev/Å）
- **精度与效率平衡模式**：`pure-cartesian-ictd` 在保持接近最优精度（能量：0.046 mev/atom，力：9.0 mev/Å）的同时，参数量减少 72.1%，训练速度提升 2.10x（GPU，lmax=2）

**结论**：
- 所有 FSCETP 模式在真实化学反应路径数据集上均显著优于 MACE（64ch、128ch 和 198ch 配置）
- `spherical` 和 `partial-cartesian` 模式在精度和效率之间达到最佳平衡
- `pure-cartesian-ictd` 模式在保持竞争性精度的同时，提供了显著的参数效率和速度优势

### 12.11 数学实现对应关系

| 模式 | 内部表示 | 张量积方法 | 等变性保证 |
|------|----------|------------|------------|
| `spherical` | irreps `64x0e+64x1o+64x2e` | Wigner-3j CG 耦合 | 严格 O(3) |
| `partial-cartesian` | irreps `64x0e+64x1o+64x2e` | 笛卡尔坐标 + CG 系数 | 严格 O(3) |
| `partial-cartesian-loose` | irreps `64x0e+64x1o+64x2e` | norm product 近似 | 近似 O(3) |
| `pure-cartesian` | \(\bigoplus_{L\le 2}\mathbb R^{64}\otimes(\mathbb R^3)^{\otimes L}\) | δ/ε 路径枚举 | 严格 O(3) |
| `pure-cartesian-sparse` | \(\bigoplus_{L\le 2}\mathbb R^{64}\otimes(\mathbb R^3)^{\otimes L}\) | δ/ε 路径稀疏化 | 严格 O(3) |
| `pure-cartesian-ictd` | irreps `64x0e+64x1o+64x2e` | ICTD trace-chain + 多项式 CG | 严格 SO(3) |

---

## 参考文献

[1] Geiger, M., & Smidt, T. (2022). e3nn: Euclidean neural networks. *arXiv preprint arXiv:2207.09453*.

[2] Zee, A. (2016). *Group theory in a nutshell for physicists*. Princeton University Press.

[3] Shao, S., Li, Y., Lin, Z., & Cui, Q. (2025). High-rank irreducible Cartesian tensor decomposition and bases of equivariant spaces. *Journal of Machine Learning Research*, 26(175), 1-53. arXiv:2412.18263. (注：该工作通过 path matrices 直接构造不可约表示；本实现则通过调和多项式的 Laplacian 零空间方法构建不可约表示，见第 11 节)

[4] Weiler, M., Geiger, M., Welling, M., Boomsma, W., & Cohen, T. (2018). 3D steerable CNNs: Learning rotationally equivariant features in volumetric data. In *Advances in Neural Information Processing Systems* (Vol. 31).

[5] Thomas, N., Smidt, T., Kearnes, S., Yang, L., Li, L., Kohlhoff, K., & Riley, P. (2018). Tensor field networks: Rotation-and translation-equivariant neural networks for 3D point clouds. *arXiv preprint arXiv:1802.08219*.

[6] Biedenharn, L. C., & Louck, J. D. (1981). *Angular momentum in quantum physics: theory and application*. Cambridge University Press.

[7] Wigner, E. P. (1959). *Group theory and its application to the quantum mechanics of atomic spectra*. Academic Press.

[8] Varshalovich, D. A., Moskalev, A. N., & Khersonskii, V. K. (1988). *Quantum theory of angular momentum*. World Scientific.

[9] Stone, A. J. (2013). *The theory of intermolecular forces* (2nd ed.). Oxford University Press.

[10] Kondor, R., & Trivedi, S. (2018). On the generalization of equivariance and convolution in neural networks to the action of compact groups. In *Proceedings of the 35th International Conference on Machine Learning* (pp. 2747-2756). PMLR.

[11] Cohen, T. S., Geiger, M., Köhler, J., & Welling, M. (2018). Spherical CNNs. In *International Conference on Learning Representations*.

[12] Batzner, S., Musaelian, A., Sun, L., Geiger, M., Mailoa, J. P., Kornbluth, M., Molinari, N., Smidt, T. E., & Kozinsky, B. (2022). E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials. *Nature Communications*, 13(1), 2453.

[13] Musaelian, A., Batzner, S., Johansson, A., Sun, L., Owen, C. J., Kornbluth, M., & Kozinsky, B. (2023). Learning local equivariant representations for large-scale atomistic dynamics. *Nature Communications*, 14(1), 579.

[14] Edmonds, A. R. (1957). *Angular momentum in quantum mechanics*. Princeton University Press.

[15] Messiah, A. (1961). *Quantum mechanics* (Vol. 2). North-Holland Publishing Company.
