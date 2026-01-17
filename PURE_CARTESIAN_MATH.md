## Pure-Cartesian Tensor Product的数学描述

本文的主线是：在**全笛卡尔 \(3^L\)** 表示下，用 **Kronecker δ** 与 **Levi-Civita ε** 的指标代数构造**严格 O(3) 等变**的张量积；并进一步解释它与 ICTD/trace-chain（\((2\ell+1)\) irreps）之间的关系，以及本实现提供的“无球谐 irreps 内部表示”版本。


- **第 1–7 节**：只讨论 `pure-cartesian` / `pure-cartesian-sparse` 的 **严格 O(3)** 版本：内部表示是 \(\bigoplus_L (\mathbb R^3)^{\otimes L}\)（维度 \(3^L\)），并显式 true/pseudo（\(\mathbb Z_2\) 分级）来处理 \(\det(R)\)。
- **第 8 节**：解释工程里常见 irreps（如 `64x0e+64x1o+64x2e`）与 pure-cartesian（\(3^L\)）在“张量操作语义”上的对照。
- **第 9 节**：描述 `ictd_fast.py` 的“快速不可约投影”（主要面向**对称子空间**的 trace-chain / STF），它是从（通常对称的）笛卡尔张量中提取 \((2\ell+1)\) 块坐标的工具，属于 **SO(3)** 讨论范畴（默认不处理显式 pseudo 分级）。
- **第 10 节**：描述 `ictd_irreps.py` / `pure_cartesian_ictd_layers.py` 的 **ICTD-irreps 内部表示**：网络内部全程在 \((2\ell+1)\) irreps 空间做消息传递与张量积；它在 **SO(3)** 下严格等变（实现中不显式追踪 \(\det(R)\) 分级）。

对应实现：
- `molecular_force_field/models/pure_cartesian.py`
  - `PureCartesianTensorProductO3`
  - `PureCartesianTensorProductO3Sparse`
  - `_enumerate_paths`, `_enumerate_paths_sparse`, `_einsum_for_path`
  - `split_by_rank_o3`, `merge_by_rank_o3`, `rotate_rank_tensor`, `edge_rank_powers`
- `molecular_force_field/models/pure_cartesian_layers.py` / `pure_cartesian_sparse_layers.py`
  - 以消息传递方式使用上述张量积

### 阅读导航（快速定位）
- **想看严格 O(3)（含宇称）纯笛卡尔张量积**：第 1–6 节（尤其第 5–6 节）
- **想知道 sparse 到底稀疏了什么、`k_policy/allow_epsilon/assume_pseudo_zero` 的数学含义**：第 6 节
- **想理解 `64x0e+64x1o+64x2e` 与 \(3^L\) 的关系**：第 8 节
- **想看“快速 ICTD 投影”（从对称张量抽 irreps 块）**：第 9 节
- **想看“无球谐 irreps 内部表示 + 通过多项式构造 CG 张量”**：第 10 节

### 符号约定（贯穿全文）
- **\(L\)**：pure-cartesian 的“秩/阶”（rank），对应全笛卡尔表示 \((\mathbb R^3)^{\otimes L}\)，维度 \(3^L\)。
- **\(\ell\)**：不可约表示/调和多项式的“阶”（irrep degree / harmonic degree），对应维度 \(2\ell+1\) 的 irreps 块。
- **\(s\in\{0,1\}\)**：\(\mathbb Z_2\) 分级（true/pseudo）。这是严格 \(O(3)\)（含反射）所需的 bookkeeping（见第 2 节）。
- **\(R\in O(3)\)**：正交变换；\(\det(R)\in\{\pm1\}\)。当只写 \(R\in SO(3)\) 时表示仅讨论旋转（\(\det=+1\)）。

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
为了在 \(O(3)\)（而非仅 \(SO(3)\)）下严格等变，引入一个 \(\mathbb Z_2\) 分级：
- \(s=0\)：true tensor（极张量 / polar tensor）
- \(s=1\)：pseudotensor（轴张量 / axial tensor）

定义对分级块 \(T^{(s)}\in\mathcal T_L\) 的群作用为
\[
(R\cdot T^{(s)}) := (\det R)^s\,(R\cdot T).
\]
这等价于实现中 `rotate_rank_tensor(..., pseudo=s)` 的含义。

### 2.3 全特征空间（本层的输入/输出空间）
给定 \(C, L_{\max}\)，定义
\[
\mathcal V(C,L_{\max})
=\bigoplus_{s\in\{0,1\}}\ \bigoplus_{L=0}^{L_{\max}}\left(\mathbb R^C\otimes\mathcal T_L\right).
\]
实现中用 `split_by_rank_o3/merge_by_rank_o3` 将扁平向量打包/拆包成这些块。

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
因此一次 ε 收缩会引入一个额外的 \(\det R\) 因子（这是 pseudo 的来源）。

一个等价的具体定义是：
\[
\varepsilon_{ijk}=
\begin{cases}
 +1,& (i,j,k)\ \text{为}\ (1,2,3)\ \text{的偶置换}\\
 -1,& (i,j,k)\ \text{为}\ (1,2,3)\ \text{的奇置换}\\
 0,& \text{否则}.
\end{cases}
\]

实现中 `epsilon_tensor()` 即 \(\varepsilon\)。

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

### 4.1 规范化（canonical）收缩顺序（与实现一致）
- 先把第一个输入张量的“尾部” \(k\) 个指标与第二个输入张量的“尾部” \(k\) 个指标两两用 δ 收缩（实现 `_einsum_for_path` 将这些指标重命名为同一个符号实现求和）。
- 若 `use_epsilon=True`：在完成 \(k\) 次 δ 收缩后，分别从第一个张量剩余的自由指标中选择一个、从第二个张量剩余的自由指标中选择一个，与 \(\varepsilon_{ijk}\) 做一次收缩并产生一个新的输出指标。**实现中**采取的规范选择是“各自最后一个自由指标”，并把 ε 的新指标追加到输出指标串末尾。

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
也就是说，ε 路径额外产生一因子 \(\det R\)。

### 4.3 一个具体路径例子（δ 收缩）

取 \(L_1=2\)（矩阵）与 \(L_2=1\)（向量），并选 \(k=1\)、`use_epsilon=False`。则
\[
L_{\text{out}} = 2+1-2\cdot 1 = 1,
\]
其收缩对应（按某个固定的指标顺序）就是一次缩并：
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
实现里就是每条 path 配 4 份权重（对应 4 个 \((s_1,s_2)\) 组合）。

### 5.2 输出分级规则（严格宇称 bookkeeping）
定义
\[
s_{\text{out}} = s_1\oplus s_2\oplus \mathbf 1_{\text{use\_epsilon}},
\]
其中 \(\oplus\) 是 XOR。其含义：
- true × true 经 δ 仍为 true
- ε 会引入一个 \(\det R\)，相当于把 true/pseudo 翻转一次
- pseudo 的 \((\det R)\) 因子通过 XOR 规则在输出端精确传递

实现里对应：`s_out = s1 ^ s2 ^ (1 if p.use_epsilon else 0)`。

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

`PureCartesianTensorProductO3Sparse` 与上节完全相同，只是把路径集合从“全枚举”裁剪为稀疏子集 \(\mathcal P_{\text{sparse}}\)，因此仍严格 \(O(3)\) 等变（等变算子之和取子集仍等变）。

### 6.1 稀疏路径集合（实现 `_enumerate_paths_sparse`）
设 `max_rank_other`（默认 1）。仅保留满足
\[
\min(L_1,L_2)\le \texttt{max\_rank\_other}
\]
的交互：即至少一侧的秩不超过 `max_rank_other`，从而每次交互中至少有一个低秩张量（例如标量/向量；若你把 `max_rank_other` 设为 2，则也允许包含二阶张量）以减少计算。

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

### 6.5 `assume_pseudo_zero` + `allow_epsilon=False` 的严格含义
当 `assume_pseudo_zero=True` 且 `allow_epsilon=False`，实现只计算 true×true→true：
\[
s_1=s_2=0,\quad s_{\text{out}}=0,
\]
并省略所有 pseudo 通道与权重项。此时算子仍是严格 O(3) 等变（因为：
- 只用 δ 收缩，\(\Gamma_p\) 在 O(3) 下严格等变且不产生 \(\det R\)；
- 输出限定在 true 子空间仍保持等变性）。

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

## 9. 如何“操作张量”：从 `64x0e + 64x1o + 64x2e` 到 pure-cartesian 的对照

这一节解释工程上常见的默认 irreps（例如 `64x0e + 64x1o + 64x2e`）在“张量操作”意义下是什么，以及它与本文档的 pure-cartesian（\(3^L\)）表示之间如何对照。

### 10.1 `64x0e + 64x1o + 64x2e` 的块结构（irreps 语义）

这是 **不可约表示（irreps）** 的直和，表示一个特征向量按 \((\ell,p)\) 分块：
- \(64\times 0e\)：64 个**标量**通道（\(\ell=0\)，even parity）
- \(64\times 1o\)：64 个**极向量**通道（\(\ell=1\)，odd parity），每个通道 3 个分量
- \(64\times 2e\)：64 个**二阶不可约张量**通道（\(\ell=2\)，even parity），每个通道 5 个分量（STF 基）

因此总维度是
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

这里 \(\mathbb R^{3}\) 与 \(\mathbb R^{5}\) 不是“任意 3 或 5 维向量”，而是 **O(3)/SO(3) 的不可约表示空间**（带固定的基与 \(D^\ell(R)\) 作用）。

### 10.2 irreps 张量积在做什么：CG（Wigner-3j）耦合 + 通道混合

在 irreps 语义里，“张量积/双线性相互作用”不是对笛卡尔分量做随意乘法，而是对每条允许耦合路径
\[
(\ell_1,p_1)\otimes(\ell_2,p_2)\to(\ell_{\text{out}},p_{\text{out}})
\]
用 CG 系数把 \(m\) 指标耦合成 \(\ell_{\text{out}}\) 的分量，再做通道混合。其选择规则：

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
其中 \(C\) 是 Clebsch–Gordan（CG）系数。许多实现内部用 Wigner-3j 符号来生成 \(C\)；两者满足（相位约定为 Condon–Shortley 时）：
\[
C^{\ell_3,m_3}_{\ell_1,m_1;\ \ell_2,m_2}
=(-1)^{\ell_1-\ell_2+m_3}\sqrt{2\ell_3+1}
\begin{pmatrix}
\ell_1 & \ell_2 & \ell_3\\
m_1 & m_2 & -m_3
\end{pmatrix}.
\]
然后对通道 \((a,b)\) 到输出通道 \(c\) 再乘以权重并求和（与你在 `EquivariantTensorProduct` 里看到的 `W[c,a,b]` 一致）。

一个直观例子：\(\ell=1\) 与 \(\ell=1\) 的乘积允许
\[
1\otimes 1 \to 0\oplus 1\oplus 2,
\]
而宇称会决定这些输出是 \(0e/1e/2e\) 还是 \(0o/1o/2o\)（取决于输入的 \(p_1p_2\)）。

### 10.3 pure-cartesian 的“张量”是什么：\(3^L\) 秩张量而非 \(2\ell+1\) irreps

pure-cartesian 改用
\[
\mathcal T_L=(\mathbb R^3)^{\otimes L}\quad(\dim=3^L)
\]
作为秩 \(L\) 的表示空间，因此对于 `channels=64, Lmax=2`，其块是：
- rank 0：\(\mathbb R^{64}\otimes \mathcal T_0 \cong \mathbb R^{64}\)（1 分量）
- rank 1：\(\mathbb R^{64}\otimes \mathcal T_1 \cong \mathbb R^{64}\otimes\mathbb R^3\)（3 分量）
- rank 2：\(\mathbb R^{64}\otimes \mathcal T_2 \cong \mathbb R^{64}\otimes(\mathbb R^3\otimes\mathbb R^3)\)（9 分量）

与 irreps 的关键差异在于：rank 2 在这里是“全 \(3\times 3\)”（9 分量），而 irreps 的 \(\ell=2\) 是其中的 **对称无迹（STF）** 子空间（5 分量）。

### 10.4 pure-cartesian 里如何“做张量积”：只允许 δ/ε 外积与收缩

pure-cartesian 的基本双线性算子由路径 \(p\) 定义（第 4 节）：
- **外积**：把两个张量拼起来，秩相加
- **δ 收缩**：用 \(\delta_{ij}\) 成对消去一个来自左侧、一个来自右侧的指标（秩减少 2）
- **ε 收缩**：用 \(\varepsilon_{ijk}\) 消去左右各一个指标，并产生一个新指标（秩减少 1），同时引入 \(\det R\)

也就是说，你的“张量操作”就是在指标层面允许这三类操作的组合；实现通过 `_einsum_for_path` 生成规范化的 Einstein 求和式来执行。

### 10.5 为什么 pure-cartesian 需要 true/pseudo：严格 O(3)（含反射）等变

因为 \(\varepsilon\) 在 \(O(3)\) 下满足
\[
R^{\otimes 3}\varepsilon = (\det R)\varepsilon,
\]
一次 ε 路径会多出 \(\det R\)。为了让输出在 \(O(3)\) 下仍按某个表示空间闭合，需要把特征空间分成 true/pseudo 两类，并用
\[
s_{\text{out}} = s_1\oplus s_2\oplus \mathbf 1_{\varepsilon}
\]
来追踪 \(\det R\) 因子（对应 `PureCartesianTensorProductO3` 的实现）。

### 10.6 总结：两种“张量操作”语义的对照

- irreps（例如 `64x0e+64x1o+64x2e`）：
  - 张量积 = CG（Wigner-3j）耦合 \(m\) 指标 + 通道混合
  - \(\ell\) 是不可约阶数，块维度是 \(2\ell+1\)
  - \(\ell=2\) 是 STF（5 分量）
- pure-cartesian：
  - 张量积 = δ/ε 的指标外积与收缩（路径求和）+ 通道混合
  - rank \(L\) 块维度是 \(3^L\)
  - rank 2 是全 \(3\times 3\)（9 分量），不会先投影到 STF

---

## 10. 快速做不可约分解：L≤2 闭式 + 3≤L≤6 预计算 trace 链投影

本实现包含一个纯笛卡尔的“快速不可约分解”模块（无球谐、无 CG）：

- `molecular_force_field/models/ictd_fast.py`
  - `decompose_rank2_generic(T)`：对 **generic rank-2** 张量给出经典 ICTD 分解
    \[
    T_{ij}=\underbrace{\Big(\tfrac12(T_{ij}+T_{ji})-\tfrac13\delta_{ij}T_{kk}\Big)}_{\ell=2\ \text{(STF)}}
    +\underbrace{\tfrac12(T_{ij}-T_{ji})}_{\ell=1\ \text{(antisym)}}
    +\underbrace{\tfrac13\delta_{ij}T_{kk}}_{\ell=0\ \text{(trace)}}
    \]
    并把 \(\ell=1\) 的反对称块用 \(\varepsilon_{kij}\) 映射成 3 维伪向量。
  - `FastSymmetricTraceChain(Lmax<=6)`：对 **rank-L（L≤6）** 张量，预计算整条 trace 链的线性投影
    \[
    P_{L\to \ell}:\ \mathbb R^{3^L}\to \mathbb R^{2\ell+1},\qquad
    \ell\in\{L,L-2,L-4,\dots\},
    \]
    运行时得到各个 \(\ell\) 块的坐标（更准确地说：先取 **对称部分**，再在 trace 链子空间上投影得到坐标）。
  - （保留）`FastSymmetricSTF(Lmax<=6)`：仅输出 STF（\(\ell=L\)）块，是 trace 链的特例。

### 10.1 为什么更快？

- 对每个 \(L\le 6\)，矩阵规模最多 \(P_6\in\mathbb R^{13\times 729}\)，非常小；
- 运行时只做一次 `matmul/einsum`，无需显式对称化（已被吸收到预计算矩阵里）；
- 这条路线特别适合“几何张量通常对称”（例如 \(n^{\otimes L}\)）的场景。

### 10.2 数学核心（无球谐版 harmonic/STF）

本节把 `ictd_fast.py` 的实现逻辑用数学写清楚，并明确区分两种“坐标空间”：

- **full 坐标（\(3^L\)）**：把 rank-\(L\) 张量 \(T_{i_1\dots i_L}\) 全部分量扁平化为 \(\mathrm{vec}(T)\in\mathbb R^{3^L}\)。
- **symmetric-count 坐标（\(\binom{L+2}{2}\)）**：用“计数三元组” \((a,b,c)\) 表示一个指标多重集，维度为
  \[
  D_L=\dim(\mathrm{Sym}^L(\mathbb R^3))=\binom{L+2}{2}.
  \]
  `ictd_fast.py` 先用一个线性算子 \(S_{\text{sum}}\in\mathbb R^{D_L\times 3^L}\) 把 full 坐标汇总为
  \[
  s_{abc}=\sum_{\text{counts}(i_1,\dots,i_L)=(a,b,c)} T_{i_1\dots i_L},
  \]
  这是对称张量的自然压缩表示（对非对称张量，相当于“取对称部分的计数汇总”）。

#### 10.2.1 对称 rank-\(L\) 张量 \(\leftrightarrow\) 三元齐次多项式（次数 \(L\)）的同构

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
由于对称性只关心“每个坐标出现了多少次”，\((a,b,c)\) 唯一确定了一类（轨道）指标序列，因此可以唯一构造一个对称张量 \(T\) 使得 \(p=\Phi(T)\)。在实现里常用两套等价坐标（用来解释组合因子与 Gram 权重的来源）：

- **对称轨道平均系数** \(t_{abc}\)：把对称张量在该轨道的公共分量记为 \(t_{abc}\)（对称性保证轨道内所有 \(T_{i_1\dots i_L}\) 相等）。
- **对称轨道求和系数** \(s_{abc}\)：把该轨道内所有分量求和
  \[
  s_{abc}:=\sum_{\text{counts}(i_1,\dots,i_L)=(a,b,c)} T_{i_1\dots i_L}.
  \]
  两者满足
  \[
  s_{abc}=w_{abc}\,t_{abc},\qquad
  w_{abc}=\frac{L!}{a!\,b!\,c!}.
  \]
  因此也可写成
  \[
  t_{abc}=\frac{1}{w_{abc}}\sum_{\text{counts}(i_1,\dots,i_L)=(a,b,c)} T_{i_1\dots i_L}.
  \]

并且两边维度一致：
\[
\dim\mathrm{Sym}^L(\mathbb R^3)=\binom{L+2}{2}=\dim\mathcal P_L,
\]
故 \(\Phi\) 是线性同构（两者等价）。

#### 10.2.2 STF（对称无迹）\(\Longleftrightarrow\) 调和多项式（\(\Delta p=0\)）

对称张量的“迹”是用 \(\delta_{ij}\) 收缩两条指标：
\[
(\mathrm{tr}\,T)_{i_3\cdots i_L}:=\sum_{j=1}^3 T_{jj\,i_3\cdots i_L}\in \mathrm{Sym}^{L-2}(V).
\]
STF 的定义就是 \(\mathrm{tr}(T)=0\)（等价于所有更高次迹也为 0）。

在多项式侧，定义 Laplacian
\[
\Delta := \partial_x^2+\partial_y^2+\partial_z^2.
\]
核心对应关系是：**Laplacian 在多项式侧对应张量侧的取迹**，更精确地
\[
\Delta p_T(x)=L(L-1)\,p_{\mathrm{tr}(T)}(x).
\]
因此：
\[
\mathrm{tr}(T)=0 \quad\Longleftrightarrow\quad \Delta p_T=0.
\]
也就是说：**STF 张量对应调和多项式**（Laplacian 为零），其空间维度为 \(2L+1\)，这正是 \(SO(3)\) 的 \(\ell=L\) irreps 维度。

**直观例子（\(L=2\)）**：若 \(T_{ij}=T_{ji}\)，则
\[
p_T(x)=\sum_{i,j}T_{ij}x_i x_j,\qquad \Delta p_T = 2\,\mathrm{tr}(T).
\]
故 \(\Delta p_T=0\iff \mathrm{tr}(T)=0\)，与 rank-2 STF 条件完全一致。

对称 rank-\(L\) 张量与三元齐次多项式（次数 \(L\)）等价；STF 张量对应 **调和多项式**（Laplacian 为零）：
\[
\Delta f = (\partial_x^2+\partial_y^2+\partial_z^2)f = 0.
\]

因此 STF（\(\ell=L\)）子空间可通过 \(\ker(\Delta)\) 得到。`ictd_fast.py` 的关键细节是：

- 在 symmetric-count 坐标 \(s_{abc}\) 下，张量内积会引入对角 Gram 权重
  \[
  w_{abc}=\frac{L!}{a!\,b!\,c!},
  \qquad
  \langle s,s'\rangle = \sum_{abc}\frac{1}{w_{abc}}\,s_{abc}\,s'_{abc}.
  \]
  也就是说 Gram 矩阵是 \(G=\mathrm{diag}(1/w_{abc})\)。
- 先求 \(\ker(\Delta)\) 的一个基 \(B_L\in\mathbb R^{D_L\times(2L+1)}\)，再在上述 \(G\) 内积下做正交归一，使得
  \[
  B_L^T G B_L = I.
  \]
- 然后 STF 坐标就是
  \[
  y = B_L^T\,G\,s,\qquad s=S_{\text{sum}}\mathrm{vec}(T),
  \]
  合并得投影矩阵
  \[
  P_L = B_L^T\,G\,S_{\text{sum}}\in\mathbb R^{(2L+1)\times 3^L}.
  \]

> 注：第 11 节的 `ictd_irreps.py` 也会选一个 \(O(3)\) 不变 Gram 来固定 harmonic 基，但它用的是“各向同性高斯测度的 Gram”（非对角），二者都 \(O(3)\) 不变，因此得到的 irreps 基/坐标在数学上只相差一个 \((2\ell+1)\) 维的正交变换。

---

## 11. ICTD-irreps（trace-chain irreps 内部表示，**无球谐**）的描述

本节对应代码：

- `molecular_force_field/models/ictd_irreps.py`
- `molecular_force_field/models/pure_cartesian_ictd_layers.py`（当前的 `pure-cartesian-ictd`：**内部不再使用 \(3^L\)**，而是全程用 \((2\ell+1)\) 的 irreps）

### 11.0 本节与 O(3)（宇称/反射）的关系

本节的核心对象是 \(\mathcal H_\ell\)（调和多项式 / STF）与 trace-chain 分解。它们通常被表述为 **\(SO(3)\)** 的 \(\ell\) 阶不可约表示（维度 \(2\ell+1\)），但重要的是：在本实现的构造方式下，它们实际上**自然扩展为 \(O(3)\)** 的表示（因此包含宇称/反射），不需要像 pure-cartesian 的 ε 路径那样额外引入 true/pseudo 分级来“吸收 \(\det(R)\)”。

- **当前实现**（`ictd_irreps.py` / `pure_cartesian_ictd_layers.py`）保证的是：对任意 \(R\in SO(3)\)，张量积与消息传递满足严格等变。
- 若希望把它扩展为严格 \(O(3)\)（含 \(\det(R)=-1\)），原则上可以像第 2 节那样引入额外的 \(\mathbb Z_2\) 分级，把“\(\det(R)\) 因子”作为一个显式的表示标签随层传播。

#### 11.0.1 为什么这套 ICTD/调和多项式构造也满足 \(O(3)\)（含宇称）

关键点是：我们对多项式/张量的群作用不是“只在 \(SO(3)\) 定义”，而是对任意 \(R\in O(3)\) 都定义为变量代换（pullback）：
\[
(U_\ell(R)p)(x) := p(R^T x),\qquad R\in O(3).
\]
这本身就给出一个 \(O(3)\) 的表示，因为 \(U_\ell(R_1R_2)=U_\ell(R_1)\,U_\ell(R_2)\)。

接下来，ICTD 的所有构造步骤都与这个 \(O(3)\) 作用可交换（因此得到的投影/耦合都是 \(O(3)\) intertwiner）：

- **Laplacian 不变**：\(\Delta(p\circ R^T)=(\Delta p)\circ R^T\)，因此 \(\mathcal H_\ell=\ker\Delta\) 对任意 \(R\in O(3)\) 都是不变子空间。
- **trace-chain 的 \(r^2\) 不变**：\(\|R^T x\|^2=\|x\|^2\)，故乘以 \(r^{2k}\) 的提升算子与 \(U(R)\) 可交换。
- **选基/投影用的 Gram 是 \(O(3)\) 不变的**：不论是 `ictd_fast.py` 的对角权重 Gram，还是 `ictd_irreps.py` 的高斯 Gram
  \(\langle p,q\rangle_G\propto\int p(x)q(x)e^{-\|x\|^2/2}dx\)，都只依赖 \(\|x\|\)，因此对 \(O(3)\) 不变。这保证了“正交归一得到的基”“最小二乘投影 \(P_{L\to\ell}\)”与 \(U(R)\) 交换。

因此，从“多项式乘法 + trace-chain 投影”得到的耦合张量 \(C^{\ell_3}_{\ell_1\ell_2}\) 不仅是 \(SO(3)\) intertwiner，也是 \(O(3)\) intertwiner。

**宇称（parity）在这里是自动出现的**：对“全反演” \(R=-I\) 有
\[
(U_\ell(-I)p)(x)=p((-I)^T x)=p(-x)=(-1)^\ell p(x),
\]
也即 \(\ell\) 阶齐次（调和）多项式在宇称下天然带有 \((-1)^\ell\) 的符号。这与球谐函数的性质 \(Y_{\ell m}(-\hat n)=(-1)^\ell Y_{\ell m}(\hat n)\) 一致。

> 对照：pure-cartesian 的 ε 路径会额外引入 \((\det R)\) 因子（见第 3.2、5.2），因此需要显式 true/pseudo 分级 bookkeeping；而本节的 ICTD/调和多项式路线不显式使用 ε 收缩来“生成伪张量”，所以不需要额外的 \(\mathbb Z_2\) 通道来吸收 \(\det R\)。

### 11.1 记号

- 令 \(V=\mathbb{R}^3\)，\(R\in SO(3)\)。
- \(\mathrm{Sym}^L(V)\)：对称 rank-\(L\) 张量空间，维度 \(\binom{L+2}{2}\)。
- \(\mathcal{P}_L\)：三元齐次多项式（次数 \(L\)）空间，与 \(\mathrm{Sym}^L(V)\) 同构。
- \(\Delta=\partial_x^2+\partial_y^2+\partial_z^2\)：Laplacian（把次数 \(L\) 降到 \(L-2\)）。
- \(\mathcal{H}_L:=\ker(\Delta)\subset \mathcal{P}_L\)：调和（harmonic）子空间，\(\dim\mathcal{H}_L=2L+1\)。它对应 irreps 的 \(\ell=L\) 块（STF）。
- \(r^2=x^2+y^2+z^2\)。

### 11.2 对称张量 \(\leftrightarrow\) 齐次多项式

本节使用与 **10.2.1** 相同的同构：对称 rank-\(L\) 张量 \(\mathrm{Sym}^L(\mathbb R^3)\) 与次数 \(L\) 的三元齐次多项式空间 \(\mathcal P_L\) 线性等价。

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
它就是“对称无迹（STF）”子空间的多项式版本，维度为 \(2L+1\)。在实现中：

- 先在单项式系数基下写出 \(\Delta:\mathcal{P}_L\to\mathcal{P}_{L-2}\) 的矩阵；
- 取其零空间得到 \(\mathcal{H}_L\) 的一组基向量（列向量）。

### 11.4 用 **O(3) 不变**的高斯 Gram 固定基（关键点）

为了使得到的 \(D^{(\ell)}(R)\) 是一致/正交的，并避免“任意零空间基”带来的数值漂移，本实现用各向同性高斯测度定义内积：
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

设 \(B_L\in\mathbb{R}^{D_L\times(2L+1)}\) 是 \(\ker(\Delta)\) 的任意基（\(D_L=\binom{L+2}{2}\)）。我们通过在 \(G_L\) 内积下白化/正交化，得到最终使用的 harmonic 基（仍记为 \(B_L\)）满足：
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
于是 \(V_{\ell,k}:=M_{\ell,k}B_\ell\) 是嵌入到 \(\mathcal{P}_L\) 的 \((2\ell+1)\) 列基。对任意次数 \(L\) 的系数向量 \(t_L\)，我们用 \(G_L\) 的最小二乘投影取出 \(\ell\) 块坐标：
\[
c_\ell = P_{L\to \ell}\, t_L,
\quad
P_{L\to \ell}:=(V_{\ell,k}^T G_L V_{\ell,k})^{-1}V_{\ell,k}^T G_L.
\]
这里 \(c_\ell\in\mathbb{R}^{2\ell+1}\) 就是 irreps 块坐标（在我们固定的 harmonic 基下）。

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

**归一化/基的一致性说明**：这里构造的 \(Y_\ell(n)\) 是在“本实现选择的 harmonic 基 \(B_\ell\) 与内积 \(G_\ell\)”下的坐标。它与常见的（实）球谐函数基 \(Y_{\ell m}(\hat n)\) 一般**不要求逐分量相等**，但二者承载同一个 \((2\ell+1)\) 维 \(SO(3)\) 不可约表示，因此至多相差一个 \((2\ell+1)\times(2\ell+1)\) 的正交变换；更关键的是变换性质完全一致：\(Y_\ell(Rn)=D^{(\ell)}(R)Y_\ell(n)\)。

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
该构造自动满足选择规则：

- 三角不等式 \(|\ell_1-\ell_2|\le \ell_3\le \ell_1+\ell_2\)；
- \(\ell_1+\ell_2-\ell_3\) 必须为偶数（来自 \(r^{2k}\) 的 trace-chain 结构）。

因为这是在多项式表示空间里用 \(SO(3)\)-同态的投影构造出来的，所以 \(C^{\ell_3}_{\ell_1\ell_2}\) 是严格的 intertwiner（实现中已用“旋转前后 TP 交换”单元测试验证）。

**交换对称性（补充）**：由于多项式乘法满足交换律 \(p_a p_b=p_b p_a\)，因此交换输入 \((\ell_1,m_1)\leftrightarrow(\ell_2,m_2)\) 时，得到的耦合张量会对应“交换前两维”的同一个 intertwiner。若采用标准 Condon–Shortley 相位约定的 CG 系数，则常见的关系是
\[
(C^{\ell_3}_{\ell_1\ell_2})_{m_1m_2m_3}
=(-1)^{\ell_1+\ell_2-\ell_3}\,(C^{\ell_3}_{\ell_2\ell_1})_{m_2m_1m_3}.
\]
在本实现的构造里，具体符号/相位可能随所选基 \(B_\ell\) 发生整体的正交变换，但“交换输入对应一个确定的符号/基变换”这一结构保持不变。

### 11.8 可学习版本：shared weights + per-path gating（实现中的形式）

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
并对所有 \(p\) 求和得到边消息 \(m_e^{(\ell_3)}\)。由于 \(g_e(p)\) 是标量、而 \(\otimes_C\) 是 intertwiner，所以整体仍保持严格的 \(SO(3)\) 等变性。

---

## 12. 张量积模式性能对比（Benchmark 结果）

本节提供所有六种张量积实现模式的性能对比数据，包括参数量、前向传播速度和等变性验证结果。

### 12.1 测试环境

- **硬件**：CPU
- **模型配置**：`channels=64`, `lmax=2`
- **测试数据**：64 atoms, 512 edges
- **测试方法**：单次前向传播，30 次平均
- **等变性测试**：O(3) 等变性（包括宇称/反射），10 次随机旋转测试

### 12.2 性能对比总览

| 模式 | 等变性 | 参数量 | 相对参数量 | 速度 (ms) | 相对速度 | 等变性误差 |
|------|--------|--------|------------|-----------|----------|------------|
| `spherical` | ✅ 严格等变 | 6,540,634 | 100% (基准) | 50.79 | 1.00x (基准) | < 1e-6 ✅ |
| `partial-cartesian` | ✅ 严格等变 | 5,404,938 | 82.6% | 57.52 | 1.13x | < 1e-6 ✅ |
| `partial-cartesian-loose` | ⚠️ 近似等变 | 5,406,026 | 82.7% | 31.66 | **0.62x (最快)** | < 1e-6 ✅ |
| `pure-cartesian` | ✅ 严格等变 | 33,591,562 | 514.0% | 470.30 | 9.26x (最慢) | < 1e-6 ✅ |
| `pure-cartesian-sparse` | ✅ 严格等变 | 4,571,402 | 69.9% | 47.09 | 0.93x | < 1e-6 ✅ |
| `pure-cartesian-ictd` | ✅ 严格等变 | 1,734,304 | **26.5%** | 36.68 | 0.72x | < 1e-6 ✅ |

### 12.3 参数量详细对比（以 Spherical 为基准）

| 模式 | 参数量 | 相对参数量 | 参数量变化 | 说明 |
|------|--------|------------|------------|------|
| `spherical` | 6,540,634 | 100% | - | 基准模式，使用 e3nn 球谐函数 |
| `partial-cartesian` | 5,404,938 | 82.6% | **减少 17.4%** | 笛卡尔坐标 + CG 系数 |
| `partial-cartesian-loose` | 5,406,026 | 82.7% | **减少 17.3%** | 非严格等变（norm product 近似） |
| `pure-cartesian` | 33,591,562 | 514.0% | +414% | 全笛卡尔 \(3^L\) 表示，不推荐 |
| `pure-cartesian-sparse` | 4,571,402 | 69.9% | **减少 30.1%** | 稀疏纯笛卡尔（δ/ε 路径稀疏化） |
| `pure-cartesian-ictd` | 1,734,304 | **26.5%** | **减少 73.5%** | ICTD irreps 内部表示，参数量最少 |

### 12.4 速度详细对比（以 Spherical 为基准）

| 模式 | 时间 (ms) | 相对速度 | 说明 |
|------|-----------|----------|------|
| `spherical` | 50.79 | 1.00x (基准) | e3nn 优化实现 |
| `partial-cartesian` | 57.52 | 1.13x | 略慢于基准 |
| `partial-cartesian-loose` | 31.66 | **0.62x (最快)** | 使用 norm product 近似加速 |
| `pure-cartesian` | 470.30 | 9.26x (最慢) | 全 \(3^L\) 计算量大 |
| `pure-cartesian-sparse` | 47.09 | 0.93x | 接近基准速度 |
| `pure-cartesian-ictd` | 36.68 | 0.72x | 较快，且参数量最少 |

### 12.5 等变性验证

所有模式均通过 **O(3) 等变性测试**（包括宇称/反射），等变性误差 < 1e-6：

- ✅ **严格等变模式**：`spherical`, `partial-cartesian`, `pure-cartesian`, `pure-cartesian-sparse`, `pure-cartesian-ictd`
- ⚠️ **近似等变模式**：`partial-cartesian-loose`（使用 norm product 近似，理论上非严格等变，但数值测试中误差 < 1e-6）

### 12.6 关键发现

1. **参数量优化**：
   - `pure-cartesian-ictd` 参数量最少（减少 73.5%），且严格等变
   - `pure-cartesian-sparse` 参数量减少 30.1%，速度接近基准（0.93x）
   - `partial-cartesian` 参数量减少 17.4%，速度略慢（1.13x）

2. **速度优化**：
   - `partial-cartesian-loose` 速度最快（0.62x），但非严格等变
   - `pure-cartesian-ictd` 速度较快（0.72x），且参数量最少
   - `pure-cartesian-sparse` 速度接近基准（0.93x），严格等变

3. **不推荐使用**：
   - `pure-cartesian`（非稀疏）速度最慢（9.26x），参数量最大（+414%），仅用于研究目的

### 12.7 推荐使用场景

- **首次尝试**：使用 `spherical`（默认，标准 e3nn 实现）
- **内存极度受限**：使用 `pure-cartesian-ictd`（参数量减少 73.5%，速度 0.72x）
- **最佳平衡**：使用 `pure-cartesian-sparse`（参数量减少 30.1%，速度 0.93x，严格等变）
- **快速迭代**：使用 `partial-cartesian-loose`（速度最快 0.62x，但非严格等变）
- **严格等变 + 较少参数**：使用 `partial-cartesian` 或 `pure-cartesian-sparse`

### 12.8 数学实现对应关系

| 模式 | 内部表示 | 张量积方法 | 等变性保证 |
|------|----------|------------|------------|
| `spherical` | irreps `64x0e+64x1o+64x2e` | Wigner-3j CG 耦合 | 严格 O(3) |
| `partial-cartesian` | irreps `64x0e+64x1o+64x2e` | 笛卡尔坐标 + CG 系数 | 严格 O(3) |
| `partial-cartesian-loose` | irreps `64x0e+64x1o+64x2e` | norm product 近似 | 近似 O(3) |
| `pure-cartesian` | \(\bigoplus_{L\le 2}\mathbb R^{64}\otimes(\mathbb R^3)^{\otimes L}\) | δ/ε 路径枚举 | 严格 O(3) |
| `pure-cartesian-sparse` | \(\bigoplus_{L\le 2}\mathbb R^{64}\otimes(\mathbb R^3)^{\otimes L}\) | δ/ε 路径稀疏化 | 严格 O(3) |
| `pure-cartesian-ictd` | irreps `64x0e+64x1o+64x2e` | ICTD trace-chain + 多项式 CG | 严格 O(3) |

**注**：详细数学描述见本文档第 1–11 节。
