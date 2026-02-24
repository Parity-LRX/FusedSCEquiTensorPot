---
marp: true
theme: default
paginate: true
math: katex
style: |
  section { font-size: 26px; justify-content: flex-start; align-items: flex-start; padding-top: 0; }
  section.title { justify-content: center; align-items: center; padding-top: 0; }
  /* 小标题置顶：页内标题紧贴顶部 */
  section h1, section h2, section h3 { margin-top: 0; margin-bottom: 0.4em; padding-top: 0; }
  section > *:first-child { margin-top: 0; padding-top: 0; }
  /* _fit 页（表格等）：内容顶对齐，避免标题上方大片留白 */
  section.fit { justify-content: flex-start !important; align-items: flex-start !important; padding-top: 0 !important; }
  section.fit > div { padding-top: 0 !important; margin-top: 0 !important; }
  /* 图片页：留白、不爆框 */
  section.image-slide { padding: 1.8em 1.5em !important; }
  section.image-slide h1 { margin-bottom: 0.6em !important; }
  section.image-slide img { display: block; margin: 0 auto; max-width: 92%; max-height: 72vh; width: auto; height: auto; object-fit: contain; }
  /* 表格样式：三线表（论文风格） */
  table {
    font-size: 0.78em;
    border-collapse: collapse;
    width: 100%;
    border-top: 2px solid #333;
    border-bottom: 2px solid #333;
  }
  th, td {
    padding: 0.35em 0.5em;
    border: none; /* 去掉竖线与网格线 */
  }
  /* 表头下横线（中线） */
  table thead th { border-bottom: 1.5px solid #333; }
  /* 兼容某些渲染不生成 thead 的情况 */
  table tr:first-child th,
  table tr:first-child td { border-bottom: 1.5px solid #333; }
---

<!-- _class: title -->

# 跨层双线性融合与等变张量积实现：体阶可达性及六种张量积模式

---

1. **跨层融合（Cross-layer Fusion）**：动机与形式化
2. **预备定义**：局部环境、等变特征、体阶、等变双线性与不变量化
3. **主结果**：体阶上界与可达性（Theorem 1–2）、子空间扩展、与基线对照
4. **张量积实现**：Pure-Cartesian（δ/ε，严格 O(3)）、Sparse、ICTD（无球谐）、Partial-Cartesian / Partial-Cartesian-Loose
5. **实现与实验**：六种模式性能对比（lmax=2）、NEB 任务、结论、展望

---

# Part 1：跨层融合 — 动机

- **问题**：在固定深度 $L$、有限截断（$l_{\max}$、通道数、路径）下，如何扩展读出端可用的标量特征？
- **做法**：不仅用**末层同层二次** $\psi_{L,L}=\mathrm{Inv}(\mathrm{TP}(h^{(L)},h^{(L)}))$，还显式加入**跨层双线性等变特征** $s_{a,b}=\mathrm{TP}_{a,b}(h^{(a)},h^{(b)})$（$a<b$），再不变读出 $\psi_{a,b}=\mathrm{Inv}(s_{a,b})$。
- **效果**：在不增加消息传递深度的前提下，体阶上界从 $K_L$ 提升到 $K_a+K_b$（可达）。

---

# 预备定义（1）：局部环境与对称性

- **局部环境**：$\mathcal{N}(i)=\{j\neq i:\|r_{ij}\|\le r_c\}$，$r_{ij}=x_j-x_i+\text{PBC shift}$。
- **对称性**：$G=\mathrm{SO}(3)$，旋转 $R$ 作用 $r_{ij}\mapsto Rr_{ij}$；邻域置换 $j\mapsto\pi(j)$。
- **局部能量** $E_i$：对旋转与邻域置换不变。

**等变特征空间**（第 $t$ 层）：
$$
h_i^{(t)}\in \bigoplus_{l=0}^{l_{\max}}\left(\mathbb{R}^{c_l^{(t)}}\otimes \mathcal{H}_l\right),\quad
h_{i,l}^{(t)}(\{Rr_{ij}\})=(I_{c_l^{(t)}}\otimes D^{(l)}(R))\,h_{i,l}^{(t)}(\{r_{ij}\}).
$$

---

# 预备定义（2）：等变双线性与不变量化

- **等变双线性** $\mathrm{TP}:V\times V\to W$：
  $\mathrm{TP}(D_V(R)u,\,D_V(R)v)=D_W(R)\,\mathrm{TP}(u,v)$，且对 $u,v$ 分别线性。
- **不变量化** $\mathrm{Inv}:W\to\mathbb{R}^m$：$\mathrm{Inv}(D_W(R)w)=\mathrm{Inv}(w)$。
- **不变量双线性型**（$m=1$）：
  $\mathcal{I}(u,v):=\mathrm{Inv}(\mathrm{TP}(u,v))$ 满足 $\mathcal{I}(D(R)u,\,D(R)v)=\mathcal{I}(u,v)$。

**实现**：**irreps**（**ir**reducible **rep**resentation**s**，不可约表示）下用 CG 系数；pure-Cartesian 下用 δ/ε 收缩与 Gram 型 contraction。

---

# 体阶（Body order）的形式化

- **核分解定义**：标量 $f$ 体阶 $\le K$ 若存在 $\{f_S\}$（$|S|\le K$）使得
  $$
  f(\{r_{ij}\}_{j\in\mathcal{N}(i)})=\sum_{|S|\le K}f_S(\{r_{ij}\}_{j\in S}).
  $$
  $\mathrm{bo}(f)=\min\{K:\ f\text{ 满足上述表示}\}$。
- **Hoeffding-ANOVA**：$\mathcal{H}=\bigoplus_{k=0}^{n}\mathcal{H}^{(k)}$，$\mathrm{bo}(f)=\max\{k:\|P_k f\|_{L^2}>0\}$。

---

# 基本引理（体阶）

- **Lemma 1（求和闭包）**：$g=\sum_j g_j$，若 $\mathrm{bo}(g_j)\le K$ 则 $\mathrm{bo}(g)\le K$。
- **Lemma 2（双线性体阶相加）**：$\mathrm{bo}(\mathcal{T}(u,v))\le \mathrm{bo}(u)+\mathrm{bo}(v)$。
- **Lemma 3（下界）**：$\|P_K f\|_{L^2}>0\Rightarrow \mathrm{bo}(f)\ge K$。
- **Lemma 5（互不交变量乘积）**：$u\in\mathcal{H}^{(|S|)}$、$v\in\mathcal{H}^{(|T|)}$，$S\cap T=\varnothing$ $\Rightarrow$ $uv\in\mathcal{H}^{(|S|+|T|)}$。

---

# 跨层融合：定义（两层 / 任意 L 层）

- **两层**：$u=h^{(a)}\oplus h^{(b)}$，$\Phi_{\mathrm{cross}}^{(a,b)}=\mathrm{TP}^{\oplus}_{a,b}(u,u)$；标量读出 $\mathrm{Inv}(\Phi_{\mathrm{cross}}^{(a,b)})$。
- **L 层**：$h^{(\le L)}=h^{(1)}\oplus\cdots\oplus h^{(L)}$，$\Phi_{\mathrm{cross}}^{(L)}=\mathrm{TP}_{\mathrm{cross}}(h^{(\le L)},h^{(\le L)})$。
- **层对** $(a,b)$：$s_{a,b}=\mathrm{TP}_{a,b}(h^{(a)},h^{(b)})$，$\psi_{a,b}=\mathrm{Inv}(s_{a,b})$。
- **Anchor 集合**：$\mathcal{P}_{\mathrm{anchor}}^{(L)}=\{(t,L):1\le t\le L-1\}$，标量特征 $\Psi_{\mathrm{anchor}}^{(L)}=[\{\psi_{t,t}\},\{\psi_{t,L}\}]$。

---

# 主定理：体阶上界与可达性

- **Theorem 1**：$\mathrm{bo}(h^{(a)})\le K_a$、$\mathrm{bo}(h^{(b)})\le K_b$ $\Rightarrow$
  $$
  \mathrm{bo}(s_{a,b})\le K_a+K_b,\quad \mathrm{bo}(\psi_{a,b})\le K_a+K_b.
  $$
- **Theorem 2（可达性）**：在非退化条件下（互不交子集 $S,T$、纯成分函数、双线性型非退化、A2–A3），存在构造使 $\mathrm{bo}(s_{a,b})=K_a+K_b$。
- **Proposition 2**：逐层递推 $K_L\le K_0+L$；跨层加和可达 $\max_{(a,b)}(K_a+K_b)$，例如 $K_L+K_{L-1}=2K_L-1$（当 $K_t=K_0+t$ 时）。

---

<!-- _fit -->

# 与 MACE 的对照

|  | MACE | 跨层融合（本文） |
|------|------|------------------|
| 标量来源 | 同层 TP + $l=0$ 投影（coupling graph） | 同层 + **跨层** $\mathrm{TP}_{a,b}(h^{(a)},h^{(b)})$ 再 Inv |
| 体阶提升 | 同层多次 TP，$\nu(\Gamma)K_\rho$（Proposition 5） | 跨深度双线性，$K_a+K_b$（Theorem 1–2） |
| 标量子空间 | $\mathcal{V}_{\text{MACE}}$ | $\mathcal{V}_{\text{extended}}=\mathcal{V}_{\text{MACE}}+\mathrm{span}\{\psi_{a,b}\}$ |

- **Proposition 3–4**：扩展后函数类包含基线；在正交补上非零相关时，最优 $L^2$ 逼近误差严格下降。

---

# 梯度与优化（跨层融合）

- 读出含 $s=\mathcal{B}(u,u)$ 时，对 $u=h^{(\le L)}$ 的梯度含**直接项**：
  $$
  \nabla_{h^{(a)}}\mathcal{L}\big|_{\text{via }s}=P_a\big((\mathcal{B}(\cdot,u))^* g+(\mathcal{B}(u,\cdot))^* g\big).
  $$
- 与“仅末层读出”对比：基线时早期层梯度需经 $a\to L$ 链式传递，易出现尺度/衰减问题；跨层融合为每层提供直接梯度路径。

---

# Part 2：Pure-Cartesian 张量积

- **目标**：在**全笛卡尔** $3^L$ 表示下，用 **Kronecker δ** 与 **Levi-Civita ε** 构造**严格 O(3) 等变**张量积（含反射/宇称）。
- **与主文档对应**：主文档用 $\ell$ 表 Cartesian rank；附录用 $L$，对应关系 $\ell\leftrightarrow L$。$\mathcal{T}_L=(\mathbb{R}^3)^{\otimes L}$，$\dim\mathcal{T}_L=3^L$。

---

# O(3) 与 Z₂ 分级（true / pseudo）

- $O(3)=\{R: R^\top R=I,\ \det R\in\{\pm1\}\}$。
- **自然张量作用**：$(R\cdot T)_{i_1\cdots i_L}=\sum_{j_1,\ldots,j_L} R_{i_1j_1}\cdots R_{i_Lj_L}\,T_{j_1\cdots j_L}$。
- **true/pseudo**：$s=0$（极张量），$s=1$（轴张量）。定义
  $$
  (R\cdot T^{(s)}):=(\det R)^s\,(R\cdot T).
  $$
- 全特征空间：$\mathcal{V}(C,L_{\max})=\bigoplus_{s,L}\mathbb{R}^C\otimes\mathcal{T}_L$。

---

# δ 与 ε：唯一几何张量

- **Kronecker δ**：$\sum_{a,b}R_{ia}R_{jb}\delta_{ab}=\delta_{ij}$（O(3) 不变）。
- **Levi-Civita ε**：$\sum_{a,b,c}R_{ia}R_{jb}R_{kc}\varepsilon_{abc}=(\det R)\,\varepsilon_{ijk}$（一次 ε 引入 $\det R$，对应 pseudo）。
- 实现中仅用 δ、ε 做指标收缩，无需球谐/CG 系数即可保证 O(3) 等变。 

---

# 路径算子 Γ_{k,ε}

- **路径** $p=(k,\epsilon)$：$k$ 为 δ 成对收缩次数，$\epsilon\in\{0,1\}$ 为是否再用一次 ε。
- **输出秩**：$L_{\mathrm{out}}=L_1+L_2-2k-\epsilon$。
- **坐标定义**：
  - $\epsilon=0$：$\big(\Gamma_{k,0}(A,B)\big)_{...}=\sum_{a_1,\ldots,a_k}A_{...a_1\ldots a_k}\,B_{...a_1\ldots a_k}$。
  - $\epsilon=1$：多一处 $\varepsilon_{cuv}$ 收缩，产生新指标 $c$。
- **Proposition 1**：$\Gamma_{k,\epsilon}(R\cdot A,R\cdot B)=(\det R)^\epsilon\,R\cdot\Gamma_{k,\epsilon}(A,B)$。

---

# Pure-Cartesian TP：严格 O(3) 等变

- **映射**：$\mathrm{TP}:\mathcal{V}(C_1,L_{\max})\times\mathcal{V}(C_2,L_{\max})\to\mathcal{V}(C_{\mathrm{out}},L_{\max})$。
- **输出分级**：$s_{\mathrm{out}}=s_1\oplus s_2\oplus\mathbf{1}_{\mathrm{use\_epsilon}}$（XOR）。
- **分块**：对每条路径 $p$ 与 $(s_1,s_2)$，$U_p=\Gamma_p(X_{L_1}^{(s_1)},Y_{L_2}^{(s_2)})$，再通道混合 $W_p^{(s_1,s_2)}$ 得到 $Z_{L_{\mathrm{out}}}^{(s_{\mathrm{out}})}$。
- **Theorem 1**：在上述约定下 $\mathrm{TP}(R\cdot x_1,R\cdot x_2)=R\cdot\mathrm{TP}(x_1,x_2)$。

---

# Pure-Cartesian-Sparse：稀疏路径与超参数

- **与 dense 的差别**：仅将路径集合从全枚举改为稀疏子集 $\mathcal{P}_{\mathrm{sparse}}$，几何算子仍为同一族 $\Gamma_{k,\epsilon}$，故**仍严格 O(3) 等变**。
- **稀疏条件（`_enumerate_paths_sparse`）**：设 `max_rank_other`（默认 1），只保留满足
  $$
  \min(L_1,L_2)\le \texttt{max\_rank\_other}
  $$
  的 $(L_1,L_2)$ 交互——即至少一侧为低秩（标量/向量，或为 2 时允许二阶张量），以降低计算量。

---

# Pure-Cartesian-Sparse：$k_{\mathrm{policy}}$ 与 $\varepsilon$

- **$k_{\mathrm{policy}}$**（仅当 $\min(L_1,L_2)=1$，即向量-张量时）：
  - **"k0"**：只保留 $k=0$ ⇒ $L_{\mathrm{out}}=L_1+L_2$（外积，秩升高）。
  - **"k1"**：只保留 $k=1$ ⇒ $L_{\mathrm{out}}=L_1+L_2-2$（一次 δ 收缩，秩降低）。
  - **"both"**：两者都保留。
- **$\texttt{allow\_epsilon}$**：若为 False，删除所有 $\epsilon=1$ 的路径；本层仅用 δ，**不产生新的 pseudo 分量**。
- **$\texttt{share\_parity\_weights}$**：False 时每对 $(s_1,s_2)$ 一套 $W_p^{(s_1,s_2)}$（最多 4 套）；True 时仅按 $s_{\mathrm{out}}$ 共享两套 $W_p^{(0)},W_p^{(1)}$。仅为参数共享，不改变等变性。

---

# Pure-Cartesian-Sparse：$\texttt{assume\_pseudo\_zero}$

- **$\texttt{assume\_pseudo\_zero=True}$** 且 **$\texttt{allow\_epsilon=False}$** 时：只计算 **true×true→true**（$s_1=s_2=s_{\mathrm{out}}=0$），省略所有 pseudo 通道及对应权重。
- **等变性**：此时仅含 δ 收缩，$\Gamma_p$ 不引入 $\det R$，输出限制在 true 子空间仍与 O(3) 作用交换，故**仍严格 O(3) 等变**。
- 小结：稀疏化与权重共享只减少路径/参数，不改变 $\Gamma_p$ 与分级规则 ⇒ 等变性保持不变。

---

# ICTD 思路概览（无球谐的 irreps 实现）

- **ICTD** = **I**rreducible **C**artesian **T**ensor **D**ecomposition（不可约笛卡尔张量分解）。
- **目标**：在笛卡尔坐标下得到与 irreps 等价的 $(2\ell+1)$ 维块与 CG 型耦合，**不使用球谐函数**。
- **路线**：对称张量 ↔ 齐次多项式 → 调和子空间（STF）↔ $\ell$ 阶不可约表示 → 多项式乘法 + trace-chain 投影 ⇒ 耦合张量（CG 型）。
- **实现**：`ictd_fast.py`（对称张量 → irreps 块的快速投影）、`ictd_irreps.py` / `pure_cartesian_ictd_layers.py`（多项式乘法 + 投影，irreps 内部表示）。

---

# ICTD 数学描述（1）：对称张量 ↔ 齐次多项式

- **$\mathrm{Sym}^L(V)$**：对称 rank-$L$ 张量，$\dim=\binom{L+2}{2}$；**$\mathcal{P}_L$**：三元齐次多项式次数 $L$。
- **同构 $\Phi$**：$\Phi(T)(x)=T(x,\ldots,x)$；坐标形式 $\Phi(T)(x)=\sum_{i_1,\ldots,i_L}T_{i_1\cdots i_L}\,x_{i_1}\cdots x_{i_L}$。
- **逆 $\Psi$**（极化）：$\Psi(p)(u_1,\ldots,u_L)=\frac{1}{L!}\frac{\partial^L}{\partial t_1\cdots\partial t_L}\,p\bigl(\sum_r t_r u_r\bigr)\big|_{t=0}$ ⇒ $\mathrm{Sym}^L(V)\cong\mathcal{P}_L$。
- **O(3) 作用**：多项式侧 $(U_L(R)p)(x)=p(R^\top x)$，与对称张量的自然作用在 $\Phi$ 下一致。

---

# ICTD 数学描述（2）：STF ↔ 调和多项式

- **迹**：对称张量 $\mathrm{tr}\,T$ 为用 $\delta_{ij}$ 收缩两条指标；多项式侧 **Laplacian** $\Delta=\partial_x^2+\partial_y^2+\partial_z^2:\mathcal{P}_L\to\mathcal{P}_{L-2}$，满足 $\Delta\,\Phi(T)=L(L-1)\,\Phi(\mathrm{tr}\,T)$。
- **STF**：$\mathrm{tr}\,T=0$ ⇔ **调和** $\mathcal{H}_L:=\ker(\Delta)\subset\mathcal{P}_L$，$\dim\mathcal{H}_L=2L+1$（即 $\ell=L$ 的 irreps 维数）。
- **Fischer 分解**：
  $$
  \mathcal{P}_L=\mathcal{H}_L\oplus r^2\mathcal{P}_{L-2},\quad
  \mathcal{P}_L=\bigoplus_{k=0}^{\lfloor L/2\rfloor} r^{2k}\mathcal{H}_{L-2k}.
  $$
- **trace-chain**：从 $\mathcal{P}_L$ 依次剥离 $r^{2k}\mathcal{H}_{L-2k}$ 得到各 $\ell=L-2k$ 块，对应实现中的“trace-chain 投影” $P_{L\to\ell}$。

---

# ICTD 数学描述（3）：正交基与边特征 $Y_\ell(n)$

- **高斯内积**：$\langle p,q\rangle_G=\mathbb{E}_{X\sim\mathcal{N}(0,I_3)}[p(X)q(X)]$ 为 O(3) 不变；在 $\mathcal{H}_\ell$ 上取 $G$-正交归一基 $B_\ell$，诱导表示矩阵 $D^{(\ell)}(R)$。
- **边方向特征**（无球谐）：给定单位向量 $n$，$p_{n,\ell}(x)=(n\cdot x)^\ell\in\mathcal{P}_\ell$，投影到 $\mathcal{H}_\ell$ 并取基坐标：
  $$
  Y_\ell(n):=B_\ell^\top G_\ell\,t_\ell(n)\in\mathbb{R}^{2\ell+1}.
  $$
- **Proposition 6**：$Y_\ell(Rn)=D^{(\ell)}(R)Y_\ell(n)$。与常用实球谐至多差一正交基变换，变换性质一致。

---

# ICTD 数学描述（4）：CG 型耦合（多项式乘法 + 投影）

- **乘法**：$m:\mathcal{H}_{\ell_1}\times\mathcal{H}_{\ell_2}\to\mathcal{P}_{\ell_1+\ell_2}$，$(p,q)\mapsto pq$；满足 $m(U_{\ell_1}(R)p,U_{\ell_2}(R)q)=U_{\ell_1+\ell_2}(R)\,m(p,q)$。
- **trace-chain 投影** $P_{\ell_1+\ell_2\to\ell_3}$ 与 $U(R)$ 交换（因由 $G$-内积定义），故
  $$
  \mathrm{TP}_{\ell_1,\ell_2\to\ell_3}(a,b):=P_{\ell_1+\ell_2\to\ell_3}(p_a p_b)
  $$
  为 **O(3)-intertwiner 双线性**：$\mathrm{TP}(D^{(\ell_1)}(R)a,D^{(\ell_2)}(R)b)=D^{(\ell_3)}(R)\,\mathrm{TP}(a,b)$（Proposition 7）。
- **CG 张量**：在固定基下存在三阶张量 $C^{\ell_3}_{\ell_1\ell_2}$ 使 $(c_{\ell_3})_{m_3}=\sum_{m_1,m_2}a_{m_1}b_{m_2}(C^{\ell_3}_{\ell_1\ell_2})_{m_1m_2m_3}$；选择规则 $|\ell_1-\ell_2|\le\ell_3\le\ell_1+\ell_2$，$\ell_1+\ell_2-\ell_3$ 为偶数。**全程无需 Wigner-3j/球谐**，仅多项式与线性投影。

---

# ICTD：可学习版与数值等变性

- **可学习**：每条边、每条路径 $p=(\ell_1,\ell_2\to\ell_3)$ 有标量 gate $g_e(p)$，路径共享权重 $W_p\in\mathbb{R}^{\mu_{\mathrm{out}}\times\mu_1\times\mu_2}$，路径贡献为 $g_e(p)\cdot[W_p\cdot(x^{(\ell_1)}\otimes_C y^{(\ell_2)})]$，对 $p$ 求和得边消息；$\otimes_C$ 为上述 CG 型 intertwiner。
- **理论等变性**：构造为 O(3) intertwiner（乘法与 $G$-投影均与 $U(R)$ 交换），故在精确算术下严格等变。
- **数值等变误差**：实现中（正交化、最小二乘、条件数等）在典型测试下约 **$10^{-7}$** 量级；与 pure-cartesian / spherical 的机器精度级误差不同，需在应用中权衡速度/参数量与等变精度。

---

<!-- _fit -->

# Sparse vs ICTD 小结

|  | Pure-Cartesian-Sparse | ICTD（pure-cartesian-ictd） |
|------|------------------------|-----------------------------|
| 表示 | $\bigoplus_L \mathbb{R}^C\otimes(\mathbb{R}^3)^{\otimes L}$（$3^L$） | irreps 块 $\mathbb{R}^\mu\otimes\mathbb{R}^{2\ell+1}$ |
| 耦合 | δ/ε 路径 $\Gamma_{k,\epsilon}$（稀疏子集） | 多项式乘法 + trace-chain 投影（CG 型） |
| 等变性 | 严格 O(3)（精确算术） | 理论 O(3) intertwiner；数值 ~1e-7 |
| 参数量/速度 | 适中（约 70% 参数量，1.39x） | 最少/最快（约 28%，4.12x） |

---

# Partial-Cartesian：张量积模式（严格等变）

- **定义（算子层面）**：partial-cartesian 仍以 irreps（irreducible representations，不可约表示）分块为**表示空间**，并实现一个与 $O(3)$ 表示对易的双线性耦合算子（intertwiner），其作用等价于标准 irreps-TP（CG 耦合）在某种坐标/实现安排下的实现。  
- **结论（等变性）**：在精确算术下，该模式满足严格等变性
  $$
  \mathrm{TP}(\rho_{\mathrm{in}}(R)x,\rho_{\mathrm{in}}(R)y)=\rho_{\mathrm{out}}(R)\mathrm{TP}(x,y),\qquad \forall R\in O(3).
  $$
- **实现差异（与 spherical）**：差异在于中间张量的坐标组织与算子实现方式；在采用同一组 TP 路径与 CG 耦合规则的前提下，两者对应的 intertwiner 结构与可生成的耦合通道集合一致。具体计算代价取决于实现细节与硬件环境，本文不作进一步断言。  
- **实现差异（与 Pure-Cartesian）**：Pure-Cartesian 在 $3^L$ 全笛卡尔张量空间上用 δ/ε 收缩直接构造；partial-cartesian 保持 irreps 作为内部表示并在该空间内完成耦合/投影。  

---

# Partial-Cartesian-Loose：张量积模式（近似构造）

- **近似的具体形式（norm product approximation）**：用 **norm product** 近似替代严格的 CG收缩：对给定 irreps 分块 $f_1,f_2$，以其块范数构造标量近似（例如 $\|f_1\|^2\cdot\|f_2\|^2$），并用该标量驱动通道混合；当输出包含非标量块（$l_{\mathrm{out}}>0$）时，再用一个小型可学习“方向模板”将标量扩展为 $(2l_{\mathrm{out}}+1)$ 分量（不等价于 CG 系数）。因此该模式整体算子一般**不再是严格的 $O(3)$-intertwiner**。  
- **结论**：对一般输入与一般 $R\in O(3)$，不声明严格等变恒等式成立；该模式应被视为 *non-intertwiner approximation*。  
- **数值误差的解释**：在有限精度与特定测试分布下观察到的“误差较小”仅是经验指标，不能推出对所有 $R,x$ 的严格等变；需要用式
  $$
  \mathrm{err}(R,x)=\frac{\|\mathrm{TP}(\rho_{\mathrm{in}}(R)x)-\rho_{\mathrm{out}}(R)\mathrm{TP}(x)\|}{\|\mathrm{TP}(x)\|+\varepsilon}
  $$
  等统计量单独报告其数值行为。  

---

<!-- _fit -->

# Irreps 与 Pure-Cartesian 对应

| | irreps（如 64×0e+64×1o+64×2e） | pure-cartesian |
|------|----------------------------------|----------------|
| 秩 L 块 | $\ell$ 阶，维数 $2\ell+1$；$\ell=2$ 为 STF(5) | 维数 $3^L$；rank 2 为全 $3\times3$(9) |
| 张量积 | CG（Wigner-3j）耦合 + 通道混合 | δ/ε 路径收缩 + 通道混合 |
| 宇称 | e/o $\leftrightarrow$ $p=\pm1$ | true/pseudo $s_{\mathrm{out}}=s_1\oplus s_2\oplus\epsilon$ |

---

<!-- _fit -->

# 实现模式与性能（lmax=2 示意）

| 模式 | 理论等变性 | 参数量(相对) | CPU 加速比 | GPU 加速比 | 数值等变误差 |
|------|------------|--------------|------------|------------|--------------|
| spherical | O(3) 严格 | 100% | 1.00x | 1.00x | ~1e-15 |
| pure-cartesian-sparse | O(3) 严格 | 70.4% | 1.39x | 1.17x | ~1e-15 |
| pure-cartesian-ictd | 理论 O(3) | **27.9%** | **4.12x** | **2.10x** | ~1e-7 |
| partial-cartesian | O(3) 严格 | 82.6% | 1.06x | 0.75x | ~1e-14 |
| partial-cartesian-loose | 无严格保证（近似） | 82.7% | 1.33x | 1.15x | ~1e-15 |

- 加速比与参数量均为 **lmax=2** 下、相对 spherical 基准（前向+反向总训练时间）；pure-cartesian-ictd 参数量最少、CPU/GPU 加速最高；等变误差来自数值实现。

---

# 实际任务：数据集与配置

- **数据集**：五条**氮氧化物与碳结构**反应路径的 **NEB**（Nudged Elastic Band）数据；截取到力收敛阈值 **fmax:0.1~0.2**，共 **2600+** 条构型。测试集：每个反应取 1–2 条完整或不完整路径。
- **计算软件与 DFT**：**Gaussian**；DFT 方法 **B3LYP(D3)/def2-TZVP**。
- **测试配置**：**64 channels**，**lmax=2**，**float64**。对比基线为 MACE（同 lmax，64/128/198ch）；FSCETP 为跨层融合 + 多种张量积模式（spherical / partial-cartesian / pure-cartesian-sparse / pure-cartesian-ictd 等）。

---

<!-- _class: image-slide fit -->

# 反应路径

<img src="neb.png" alt="反应路径">

---

<!-- _fit -->

# 实际任务：能量与力 RMSE 对比

| 方法 | 配置 | 模式 | 能量 RMSE (mev/atom) | 力 RMSE (mev/Å) |
|------|------|------|---------------------|-----------------|
| MACE | Lmax=2, 64ch | - | 0.13 | 11.6 |
| MACE | Lmax=2, 128ch | - | 0.12 | 11.3 |
| MACE | Lmax=2, 198ch | - | 0.24 | 15.1 |
| FSCETP | Lmax=2, 64ch | spherical | **0.044** ⭐ | 7.4 |
| FSCETP | Lmax=2, 64ch | partial-cartesian | 0.045 | 7.4 |
| FSCETP | Lmax=2, 64ch | partial-cartesian-loose | 0.048 | 8.4 |
| FSCETP | Lmax=2, 64ch | **pure-cartesian-sparse** | **0.044** ⭐ | **6.5** ⭐ |
| FSCETP | Lmax=2, 64ch | pure-cartesian-ictd | 0.046 | 9.0 |

数值越小越好；⭐ 表示该列最佳。

---

# 实际任务：定量对比与模式

- **相对 MACE(64ch)**：FSCETP 能量 RMSE 约 **降低 66.2%**（0.044 vs 0.13 mev/atom）；力 RMSE 约 **降低 44.0%**（6.5 vs 11.6 mev/Å）。相对误差比约 0.34（能量）、0.56（力）。
- **FSCETP 各模式**（本配置下）：**能量最优**：`spherical` 与 `pure-cartesian-sparse` 并列（0.044）；**力最优**：`pure-cartesian-sparse`（6.5）；**效率与精度折中**：`pure-cartesian-ictd`（0.046 / 9.0，参数量约 28%、加速约 4.12x）。
- **结论**：所有 FSCETP 模式在本 NEB 数据上均显著优于 MACE（64/128/198ch）；跨层融合 + 多种等变张量积实现均能带来精度与/或效率提升。

---

# 小结（1）

- **跨层融合（主贡献）**：在固定消息传递深度与有限截断配置下，于读出端引入跨深度双线性等变特征并不变量化 $\psi_{a,b}=\mathrm{Inv}(\mathrm{TP}_{a,b}(h^{(a)},h^{(b)}))$，从而系统性扩展标量特征生成集合。  
- **体阶刻画与可达性**：跨层双线性不变量满足体阶上界 $\mathrm{bo}(\psi_{a,b})\le K_a+K_b$，并在非退化条件下可达 $\mathrm{bo}(\psi_{a,b})=K_a+K_b$；该结论给出在不加深网络时提升体阶的可验证机制。  
- **表示能力（子空间扩展）**：加入跨层生成元后，读出端特征空间由 $\mathcal{W}_{\mathrm{base}}$ 扩展为 $\mathcal{W}_{\mathrm{base}}+\mathrm{span}(\mathcal{G}_{\mathrm{cross}})$，因此最优 $L^2$ 逼近误差单调不增，并在适当条件下严格下降。  

---

# 小结（2）

- **等变张量积实现（方法学贡献）**：在统一框架下给出多种张量积实现以覆盖“严格等变—近似等变—参数-不同坐标系的谱系：  
  - **spherical / partial-cartesian**：基于 irreps 的 CG/intertwiner 实现，严格 $O(3)$ 等变（精确算术）。  
  - **partial-cartesian-loose**：采用 **norm product approximation** 等近似替代严格 CG 收缩，整体一般为 non-intertwiner（无严格等变保证）。  
  - **pure-cartesian / pure-cartesian-sparse**：在 $3^L$ 笛卡尔张量空间上以 δ/ε 收缩并配合 true/pseudo 分级，严格 $O(3)$ 等变。  
  - **pure-cartesian-ictd（ICTD）**：通过调和多项式与 trace-chain 投影构造 irreps 耦合（无球谐），理论上为 intertwiner；数值实现可观察到非零等变误差。  
- **实验验证**：在 NEB 数据（lmax=2, 64ch）上，FSCETP 各模式在能量/力 RMSE 上均优于基线；不同张量积模式在精度与参数量/速度上呈现可区分的权衡，与“读出端特征子空间扩展”及“多种等变/近似实现可选”的理论设定相一致。  

---

# 未来展望：Diffusion 生成 + FSCETP 解码器的低能垒材料搜索

- **核心想法**：用扩散生成模型提出候选“材料/结构/表面位点/吸附构型”，用 **FSCETP** 作为 **decoder / scorer** 快速评估关键反应步骤的能垒与反应能，从而在大规模候选空间中筛选**最低能垒**体系。
- **工作流**：
  - **生成**：Diffusion 在结构空间采样候选（晶体/团簇/表面 + 吸附物与位点），并支持约束（元素组成、对称性、密度、价态、位点类型等）。
  - **解码/打分**：FSCETP 预测候选体系在关键构型上的能量与力；结合 NEB/TS 搜索的少量几何优化，给出近似 **$E^\ddagger$**（能垒）与 $\Delta E$（反应热）。
  - **搜索目标**：以 $\min E^\ddagger$ 为主目标，并加入约束/惩罚项（结构稳定性、可合成性、吸附能窗口、元素成本等）。
  - **主动学习**：对不确定度高或得分最优的候选做 DFT 复核 → 反向加入训练集 → 迭代提升 FSCETP 在目标反应家族上的泛化。
- **预期收益**：把高精度但昂贵的 DFT/NEB前置为少量验证，将大规模探索交给“生成 + 等变力场解码”，实现更快的**反应能垒最小化**材料发现。 
