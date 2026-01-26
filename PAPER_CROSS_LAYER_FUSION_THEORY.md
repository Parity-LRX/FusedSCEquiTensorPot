# 跨层融合（Cross-layer Fusion）的数学分析

> 本文给出跨层融合（cross-layer fusion）结构的形式化分析。在固定的模型截断配置与资源约束（例如角动量截断、通道数与张量积路径集合受限、以及有限深度）下，本文比较不同等变势能模型构造不变量特征子空间的方式，并给出跨层双线性交叉不变量对可生成不变量子空间与体阶可达性的理论刻画。

---

## 1. 预备定义与假设（Preliminaries）

### Definition 1（局部环境）
固定中心原子 \(i\)。其截断邻域记为
\[
\mathcal{N}(i)=\{j\neq i:\|r_{ij}\|\le r_c\},\qquad r_{ij}=x_j-x_i+\text{PBC shift}.
\]

### Definition 2（对称性作用）
令 \(G=\mathrm{SO}(3)\)。旋转 \(R\in G\) 作用为 \(r_{ij}\mapsto Rr_{ij}\)。邻域置换群 \(S_{\mathcal{N}(i)}\) 作用为索引重排 \(j\mapsto \pi(j)\)。

局部能量函数 \(E_i\) 满足：
\[
E_i(\{Rr_{ij}\}_{j})=E_i(\{r_{ij}\}_j),\qquad
E_i(\{r_{i\pi(j)}\}_j)=E_i(\{r_{ij}\}_j).
\]

### Definition 3（等变特征空间）
对 \(l=0,\dots,l_{\max}\)，记 \(\mathcal{H}_l\) 为维度 \(2l+1\) 的不可约表示空间，表示矩阵为 \(D^{(l)}(R)\)。
第 \(t\) 层在中心 \(i\) 的等变特征属于
\[
h_i^{(t)}\in \bigoplus_{l=0}^{l_{\max}}\left(\mathbb{R}^{c_l^{(t)}}\otimes \mathcal{H}_l\right),
\]
其中 \(c_l^{(t)}\) 表示第 \(t\) 层在角动量 \(l\) 的通道数（multiplicity）。并满足等变性：
\[
h_{i,l}^{(t)}(\{Rr_{ij}\}_j)=(I_{c_l^{(t)}}\otimes D^{(l)}(R))\,h_{i,l}^{(t)}(\{r_{ij}\}_j),
\]
其中 \(I_{c_l^{(t)}}\) 表示 \(c_l^{(t)} \times c_l^{(t)}\) 的单位矩阵。

> 说明：e3nn 的不可约表示（irreps）实现、Cartesian-irreps、ICTD-irreps 以及 pure-Cartesian（rank-\(\ell\) Cartesian 张量，其坐标维度为 \(3^{\ell}\)）实现均可置于上述抽象框架中；差异仅在于表示空间的具体坐标系与张量积的实现形式。为避免与“网络深度 \(L\)”混淆，本文用 \(\ell\) 表示张量阶（rank）。

### Definition 4（等变双线性张量积与不变量算子）
记 \(D(R)\) 为直和空间上的群表示。称算子 \(\mathrm{TP}(\cdot,\cdot)\) 为**等变双线性**，若
\[
\mathrm{TP}(D(R)u,\ D(R)v)=D(R)\mathrm{TP}(u,v),\quad
\mathrm{TP}(\alpha u_1+\beta u_2,v)=\alpha \mathrm{TP}(u_1,v)+\beta \mathrm{TP}(u_2,v),
\]
并对第二自变量同理。

进一步，称 \(\mathcal{I}(u,v)\) 为**不变量双线性型**，若
\[
\mathcal{I}(D(R)u,\ D(R)v)=\mathcal{I}(u,v),\qquad \forall R\in G.
\]

在不可约表示（irreps）表述下，\(\mathcal{I}\) 可由“等变张量积后投影到标量表示（\(l=0\)）”给出；在 pure-Cartesian 表述下，\(\mathcal{I}\) 可由 δ/ε 收缩得到的标量（或 Gram 型 contraction）实现；在 ICTD 表述下，\(\mathcal{I}\) 可由对 \(m\) 指标的收缩（contraction）实现。

**假设 A1（实现层面的可分解性）**：本文讨论的 \(\mathcal{I}\) 进一步满足如下可分解形式：存在有限 \(P<\infty\) 以及线性算子 \(A_p,B_p\)，使得对任意向量值特征 \(u,v\) 有
\[
\mathcal{I}(u,v)=\sum_{p=1}^{P}\langle A_p u,\ B_p v\rangle,
\]
其中 \(\langle\cdot,\cdot\rangle\) 表示坐标上的点乘（等价于有限项的"坐标乘积后求和"）。该假设对本文涉及的主流不变量实现成立：
- **在不可约表示（irreps）表述下的收缩（contraction）**：对 \(m\) 指标求和，为有限个坐标乘积的线性组合
- **Gram 型 contraction**：对张量指标求和（如 \(\sum_{\alpha,\beta} u_\alpha v_\beta \delta_{\alpha\beta}\)），为有限项点积
- **pure-Cartesian 中的 δ/ε contraction**：爱因斯坦求和约定下的有限指标收缩，等价于有限项乘积求和
- **ICTD 表述下的 \(m\) 指标收缩（contraction）**：与 irreps 表述类似，为有限项坐标乘积的线性组合

---

## 2. 多体阶数（Body order）的形式化定义

为避免“对邻居求和是否使体阶 +1”的歧义，我们采用标准的“子集核分解”定义（与 \(N\)-body expansion / Hoeffding-ANOVA 在对称函数空间上的分解一致）。

### Definition 5（体阶 \(\mathrm{bo}(\cdot)\)）
设 \(f\) 是定义在有限多集 \(\{r_{ij}\}_{j\in\mathcal{N}(i)}\) 上的置换不变标量函数。称 \(f\) 的体阶不超过 \(K\)，若存在一族核函数 \(\{f_S\}\)（对所有 \(S\subseteq \mathcal{N}(i)\) 且 \(|S|\le K\)），使得
\[
f(\{r_{ij}\}_{j\in\mathcal{N}(i)})
=\sum_{\substack{S\subseteq \mathcal{N}(i)\\ |S|\le K}} f_S(\{r_{ij}\}_{j\in S}),
\]
其中每个 \(f_S\) 仅依赖于子集 \(S\) 中的相对位移（及其原子种类等局部属性）。
定义
\[
\mathrm{bo}(f)=\min\{K:\ f \text{ 满足上述表示}\}.
\]

> 注：此定义把“体阶”与“依赖于多少个邻居索引”的概念严格对齐；并与 ACE/MTP/MACE 文献中的 \(N\)-body 子空间一致。

### Definition 5'（\(K\)-body 子空间与正交分解：Hoeffding/ANOVA 视角）
为给出体阶“下界/可达性”的可检验判别，考虑固定最大邻居数 \(n:=|\mathcal{N}(i)|\) 的情形。设 \(\mathcal{X}\) 表示单个相对位移（及元素类型标签等）所在的样本空间，定义
\[
\mathcal{H}:=L^2_{\mathrm{sym}}(\mathcal{X}^n)
\]
为 \(\mathcal{X}^n\) 上置换对称（交换任意邻居坐标不变）的平方可积函数空间（参考测度可取由数据分布诱导的经验测度或其极限）。

**假设 A2（Hoeffding-ANOVA 正交分解存在）**：在所选参考测度下，\(\mathcal{H}\) 存在 Hoeffding-ANOVA 的正交分解
\[
\mathcal{H}=\bigoplus_{k=0}^{n}\mathcal{H}^{(k)},
\]
其中 \(\mathcal{H}^{(k)}\) 是“纯 \(k\)-体成分”的闭子空间（与所有低于 \(k\) 体成分正交）。记 \(P_k:\mathcal{H}\to\mathcal{H}^{(k)}\) 为正交投影。
在该框架下，体阶可等价刻画为
\[
\mathrm{bo}(f)=\max\{k:\ \|P_k f\|_{L^2}>0\}.
\]

> 备注（投影的显式形式）：在常见的（交换）乘积测度设定下，Hoeffding 分量可用条件期望的包含-排除公式给出。令 \(f\in L^2(\mathcal{X}^n)\)，对任意索引子集 \(S\subseteq\{1,\dots,n\}\)，记 \(\mathbb{E}[f\mid S]\) 为在固定坐标 \(x_S\) 条件下、对其余坐标积分的条件期望。定义
> \[
> f_S^{\circ}:=\sum_{U\subseteq S}(-1)^{|S|-|U|}\,\mathbb{E}[f\mid U].
> \]
> 则 \(f=\sum_{S\subseteq[n]} f_S^{\circ}\) 且不同 \(|S|\) 的分量正交；“纯 \(k\)-体子空间”可由 \(\{f_S^{\circ}:\ |S|=k\}\) 张成并闭包化得到，从而 \(P_k f=\sum_{|S|=k} f_S^{\circ}\)。

**假设 A3（乘积测度/坐标独立性）**：进一步假设参考测度在坐标上为乘积形式（或至少满足：对互不相交坐标集合 \(A,B\)，在给定任意坐标子集 \(U\) 后，\(A\setminus U\) 与 \(B\setminus U\) 条件独立）。在该假设下，对任意仅依赖于坐标集合 \(A\) 的函数 \(g\) 与仅依赖于坐标集合 \(B\) 的函数 \(h\)，以及任意 \(U\subseteq A\cup B\)，有条件期望的乘积分解
\[
\mathbb{E}[g\,h\mid U]=\mathbb{E}[g\mid U\cap A]\;\mathbb{E}[h\mid U\cap B].
\]

> **注（假设 A3 的适用范围）**：在分子体系中，由于几何约束与相互作用的存在，坐标分布一般不满足严格的乘积测度假设。本文引入 A3 的目的在于在 Hoeffding-ANOVA 框架下获得关于体阶分解与可达性的可验证充分条件。对一般非乘积测度的情形，可采用更一般的条件独立结构或相关分解工具给出类似刻画；本文不在此展开。

---

## 3. 基本引理（Lemmas）

### Lemma 1（求和闭包：置换对称求和不增体阶）
若对每个 \(j\in \mathcal{N}(i)\)，函数 \(g_j\) 满足 \(\mathrm{bo}(g_j)\le K\) 且 \(g_j\) 仅依赖于某个至多 \(K\) 个邻居组成的子集（其中可以包含索引 \(j\)），则
\[
g(\{r_{ij}\}_j):=\sum_{j\in\mathcal{N}(i)} g_j(\{r_{ik}\}_{k\in \mathcal{N}(i)})
\]
满足 \(\mathrm{bo}(g)\le K\)。

**Proof.** 由 Definition 5，对每个 \(g_j\) 存在核分解 \(g_j=\sum_{|S|\le K} (g_j)_S\)。则
\[
g=\sum_j g_j=\sum_j\sum_{|S|\le K}(g_j)_S
\]
仍是 \(|S|\le K\) 的核之和，因此 \(\mathrm{bo}(g)\le K\)。□

### Lemma 2（双线性不变量导致体阶上界相加）
设 \(u,v\) 为两个（可向量值的）置换对称特征，其每个坐标分量都满足 Definition 5 的核分解，且 \(\mathrm{bo}(u)\le K_u\)、\(\mathrm{bo}(v)\le K_v\)。
令 \(\mathcal{I}\) 为任意不变量双线性型（Definition 4）。则
\[
\mathrm{bo}(\mathcal{I}(u,v))\le K_u+K_v.
\]

**Proof.** 写 \(u=\sum_{|S|\le K_u}u_S,\ v=\sum_{|T|\le K_v}v_T\) 为核分解。双线性给出
\[
\mathcal{I}(u,v)=\sum_{|S|\le K_u}\sum_{|T|\le K_v}\mathcal{I}(u_S,v_T).
\]
每一项仅依赖于索引并集 \(S\cup T\)，其大小 \(|S\cup T|\le |S|+|T|\le K_u+K_v\)。因此 \(\mathcal{I}(u,v)\) 存在 \(|U|\le K_u+K_v\) 的核分解，结论成立。□

### Lemma 3（体阶下界判别：非零 \(K\)-body 投影蕴含体阶下界）
在 Definition 5' 的 Hilbert 空间设定下，若对某个 \(K\) 有 \(\|P_K f\|_{L^2}>0\)，则 \(\mathrm{bo}(f)\ge K\)。

**Proof.** 由 Definition 5' 的等价刻画，\(\mathrm{bo}(f)=\max\{k:\|P_k f\|_{L^2}>0\}\)。因此 \(\|P_K f\|_{L^2}>0\Rightarrow \mathrm{bo}(f)\ge K\)。□

### Lemma 3'（纯成分的等价判别：条件期望为零 \(\Leftrightarrow\) Hoeffding 纯成分）
在假设 A2–A3 的设定下，取任意非空索引集合 \(S\subseteq\{1,\dots,n\}\)。设 \(g\in L^2(\mathcal{X}^n)\) 仅依赖于坐标集合 \(S\)（即 \(g(x)=g(x_S)\)）。则以下两条等价：

1. \(g\in \mathcal{H}^{(|S|)}\) 且其支持集合为 \(S\)（即 \(g\) 为关于 \(S\) 的纯 \(|S|\)-体 Hoeffding 分量）。
2. 对所有真子集 \(U\subsetneq S\)，有 \(\mathbb{E}[g\mid U]=0\)。

并且在上述条件成立时，\(P_{|S|}g=g\)。

**Proof.**
（\(2\Rightarrow 1\)）由于 \(g\) 仅依赖于 \(S\)，其 Hoeffding 分解只可能包含 \(\{U\subseteq S\}\) 上的分量。由 Definition 5' 的显式公式（包含-排除），对任意 \(U\subseteq S\) 有
\[
g_U^{\circ}=\sum_{V\subseteq U}(-1)^{|U|-|V|}\,\mathbb{E}[g\mid V].
\]
当 \(U\subsetneq S\) 时，右端仅涉及 \(V\subsetneq S\)，由条件 \( \mathbb{E}[g\mid V]=0 \) 可知 \(g_U^{\circ}=0\)。而当 \(U=S\) 时，
\[
g_S^{\circ}=\sum_{V\subseteq S}(-1)^{|S|-|V|}\,\mathbb{E}[g\mid V]
=\mathbb{E}[g\mid S]+\sum_{V\subsetneq S}(-1)^{|S|-|V|}\,\mathbb{E}[g\mid V]
=g.
\]
因此 \(g=\sum_{U\subseteq S} g_U^{\circ}=g_S^{\circ}\in \mathcal{H}^{(|S|)}\)，并且 \(P_{|S|}g=g\)。

（\(1\Rightarrow 2\)）若 \(g\in \mathcal{H}^{(|S|)}\) 且仅依赖于 \(S\)，则其 Hoeffding 分解为 \(g=g_S^{\circ}\)。对任意 \(U\subsetneq S\)，由 \(g_S^{\circ}\) 的构造可知其满足“退化性”（degeneracy）：
\[
\mathbb{E}[g_S^{\circ}\mid U]=0,
\]
这等价于其与所有低阶子空间正交（A2），从而 \(\mathbb{E}[g\mid U]=0\)。□

### Lemma 4（Hoeffding-ANOVA：互不相交变量上的“纯成分”乘积）
在 Definition 5' 的设定下，设 \(S,T\subseteq\{1,\dots,n\}\) 且 \(S\cap T=\varnothing\)。令
\[
u\in \mathcal{H}^{(|S|)} \text{ 且 } u \text{ 仅依赖于坐标 } S,\qquad
v\in \mathcal{H}^{(|T|)} \text{ 且 } v \text{ 仅依赖于坐标 } T.
\]
则乘积 \(w:=uv\) 属于 \(\mathcal{H}^{(|S|+|T|)}\) 且仅依赖于 \(S\cup T\)，并满足
\[
P_{|S|+|T|}w=w.
\]

**Proof.** 由 Lemma 3'，对所有真子集 \(S'\subsetneq S\) 有 \(\mathbb{E}[u\mid S']=0\)，并且对所有真子集 \(T'\subsetneq T\) 有 \(\mathbb{E}[v\mid T']=0\)。

考虑 \(w=uv\)。显然 \(w\) 仅依赖于 \(S\cup T\)。取任意真子集 \(U\subsetneq S\cup T\)。由假设 A3（乘积测度/坐标独立性），有条件期望的乘积分解
\[
\mathbb{E}[w\mid U]=\mathbb{E}[u\,v\mid U]
=\mathbb{E}[u\mid U\cap S]\;\mathbb{E}[v\mid U\cap T].
\]
由于 \(U\subsetneq S\cup T\)，必有 \(U\cap S\subsetneq S\) 或 \(U\cap T\subsetneq T\)（或二者皆真）。若 \(U\cap S\subsetneq S\)，则 \(\mathbb{E}[u\mid U\cap S]=0\)；若 \(U\cap T\subsetneq T\)，则 \(\mathbb{E}[v\mid U\cap T]=0\)。从而 \(\mathbb{E}[w\mid U]=0\)。
因此 \(\mathbb{E}[w\mid U]=0\) 对所有真子集 \(U\subsetneq S\cup T\) 成立，这正是 \(w\) 属于纯 \(|S|+|T|\) 体成分的判别条件。于是 \(w\in\mathcal{H}^{(|S|+|T|)}\)，并且由于 \(w\) 已经在该子空间内，正交投影满足 \(P_{|S|+|T|}w=w\)。□

### Lemma 5（函数类包含：特征拼接不降低表示能力）
设基线模型输出 \(E=\rho(\psi)\)。扩展模型输出 \(E'=\rho'(\psi\oplus \tilde\psi)\)，其中 \(\rho'\) 与 \(\rho\) 同属足够表达的 MLP 类。则存在 \(\rho'\) 的参数使得 \(E'\equiv E\)（令 \(\rho'\) 忽略 \(\tilde\psi\)）。因此扩展模型的函数类包含基线模型的函数类。□

---

## 4. 跨层融合的主结果（Main Results）

### Definition 6（跨层融合不变量）
设两层等变特征 \(h^{(a)},h^{(b)}\)（\(a<b\)）。跨层融合构造如下不变量向量：
\[
\Psi_{a,b}:=\Big[
\mathcal{I}(h^{(a)},h^{(a)}),\
\mathcal{I}(h^{(b)},h^{(b)}),\
\mathcal{I}(h^{(a)}\oplus h^{(b)},\,h^{(a)}\oplus h^{(b)})
\Big].
\]
其中第三项包含交叉不变量 \(\mathcal{I}(h^{(a)},h^{(b)})\)（由双线性展开得到）。

### Theorem 1（体阶上界的严格提升）
若 \(\mathrm{bo}(h^{(a)})\le K_a\)、\(\mathrm{bo}(h^{(b)})\le K_b\)，则跨层交叉不变量满足
\[
\mathrm{bo}(\mathcal{I}(h^{(a)},h^{(b)}))\le K_a+K_b.
\]
从而 \(\Psi_{a,b}\) 中包含体阶上界为 \(K_a+K_b\) 的项。该交叉项可视为在固定深度与有限截断配置下对不变量特征集合的补充生成元；其可达性由 Theorem 2 给出。

**Proof.** 直接应用 Lemma 2。□

### Theorem 2（可达性：存在构造使体阶达到 \(K_a+K_b\)）
在以下非退化条件下，存在特征 \(h^{(a)},h^{(b)}\) 使得
\[
\mathrm{bo}(\mathcal{I}(h^{(a)},h^{(b)}))=K_a+K_b.
\]
**条件（充分条件）**：
1. 存在两个互不相交的邻居子集 \(S,T\subseteq \mathcal{N}(i)\) 满足 \(|S|=K_a, |T|=K_b\)
2. 存在非零标量函数 \(\phi_S\in \mathcal{H}^{(|S|)}\) 与 \(\psi_T\in \mathcal{H}^{(|T|)}\)，分别仅依赖于 \(S\) 与 \(T\)（等价地：对所有真子集 \(U\subsetneq S\) 有 \(\mathbb{E}[\phi_S\mid U]=0\)，对所有真子集 \(U\subsetneq T\) 有 \(\mathbb{E}[\psi_T\mid U]=0\)）
3. \(\mathcal{I}\) 对所考虑的有限维特征空间诱导的双线性型非退化（等价地，在某一组坐标下可写为 \( \mathcal{I}(u,v)=u^\top M v\) 且 \(M\) 可逆）
4. 假设 A2–A3 成立

**Proof.** 首先由 Lemma 2 得到体阶上界
\[
\mathrm{bo}(\mathcal{I}(h^{(a)},h^{(b)}))\le K_a+K_b.
\]

下面构造达到该上界的例子以给出下界。由于 \(\mathcal{I}\) 在有限维特征空间上诱导的双线性型非退化，存在可逆矩阵 \(M\) 使得在某一组坐标下
\[
\mathcal{I}(u,v)=u^\top M v.
\]
令 \(A:=I\)、\(B:=M\)，则 \(\mathcal{I}(u,v)=\langle Au,\ Bv\rangle\)，其中 \(\langle\cdot,\cdot\rangle\) 为标准点积，且 \(A,B\) 可逆。

取任意非零 \(\phi_S\in \mathcal{H}^{(|S|)}\)、\(\psi_T\in \mathcal{H}^{(|T|)}\)（分别仅依赖于 \(S\) 与 \(T\)），并定义向量值函数
\[
\tilde u_S(x):=\phi_S(x_S)e_1,\qquad \tilde v_T(x):=\psi_T(x_T)e_1,
\]
其中 \(e_1\) 为第一标准基向量。令
\[
u_S:=A^{-1}\tilde u_S,\qquad v_T:=B^{-1}\tilde v_T.
\]
由于 \(A^{-1},B^{-1}\) 仅作用于特征坐标且为常系数线性变换，\(u_S\)（分别 \(v_T\)）仍仅依赖于 \(S\)（分别 \(T\)），且其各坐标分量仍属于 \(\mathcal{H}^{(|S|)}\)（分别 \(\mathcal{H}^{(|T|)}\)）。
由此
\[
f:=\mathcal{I}(u_S,v_T)=\langle Au_S,\ Bv_T\rangle=\langle \tilde u_S,\tilde v_T\rangle=\phi_S(x_S)\psi_T(x_T).
\]
由 Lemma 4（并使用 A2–A3），\(f\in\mathcal{H}^{(|S|+|T|)}\) 且 \(P_{|S|+|T|}f=f\)。由于 \(\phi_S,\psi_T\) 非零，故 \(\|P_{|S|+|T|}f\|_{L^2}=\|f\|_{L^2}>0\)，从而 \(\|P_{K_a+K_b}f\|_{L^2}>0\)。
由 Lemma 3 立即得到 \(\mathrm{bo}(f)\ge K_a+K_b\)。与上界合并可得
\[
\mathrm{bo}(\mathcal{I}(h^{(a)},h^{(b)}))=\mathrm{bo}(f)=K_a+K_b.
\]
□

### Corollary 1（固定深度与有限截断配置下的不变量子空间扩展）
设基线模型仅从最终层 \(h^{(L)}\) 构造不变量 \(\psi=\mathcal{I}(h^{(L)},h^{(L)})\) 并读出 \(E=\rho(\psi)\)。
跨层融合模型增加 \(\Psi_{a,b}\) 并读出 \(E'=\rho'(\psi\oplus \Psi_{a,b})\)。
则由 Lemma 5，函数类满足 \(\mathcal{F}\subseteq \mathcal{F}'\)。并且由 Theorem 1–2，\(\mathcal{F}'\) 可在**不增加消息传递深度 \(L\)** 的前提下包含更高体阶（至 \(K_a+K_b\)）的不变量成分，从而在固定截断配置下扩展可表示的不变量成分。

---

## 5. 与 MACE 的对照：有限截断配置下的不变量生成差异

### 5.1 有限截断配置下的多体不变量生成
MACE 可表述为基于等变张量积的可学习 \(N\)-body 展开（与 ACE/MTP 的不变量代数相关）。在给定模型设定下，以下截断参数通常被固定：
- 角动量截断 \(l_{\max}\)
- 每个 \(l\) 的 multiplicity（通道数）\(\{c_l\}\)
- 允许的张量积路径集合（\(l_1\otimes l_2\to l_3\) 的子集）
- interaction blocks 的数目（深度）

因此，在给定有限截断配置下，MACE 所能生成的不变量集合可表述为某个有限维线性子空间 \(\mathcal{V}_{\text{MACE}}(l_{\max},\{c_l\},\text{paths},L)\subset \mathcal{V}_{\text{inv}}\)。当截断参数与通道数在适当意义下增大时，该子空间可在更大范围内逼近不变量函数类；本文关注固定截断配置下的不变量子空间结构差异。

### 5.1.1 MACE 的多体不变量构造机制

MACE 在单个 interaction block 内的多体不变量构造遵循以下步骤：

1. **等变特征的算子链表示（intertwiner chain）**：在固定截断配置下，令
   \[
   V^{(t)}:=\bigoplus_{l=0}^{l_{\max}}\left(\mathbb{R}^{c_l^{(t)}}\otimes \mathcal{H}_l\right)
   \]
   表示第 \(t\) 层的等变特征空间。令 \(U\) 表示用于耦合的边/径向特征所张成的（有限维）等变空间（其具体形式由径向基、球谐/方向基以及元素嵌入决定；在固定截断下可视为已定）。一次层内张量积（耦合）可抽象为一个 \(G=\mathrm{SO}(3)\) 的 intertwiner：
   \[
   T^{(t)}: V^{(t-1)}\otimes U \to V^{(t)}.
   \]
   在不可约表示分量上，\(T^{(t)}\) 由 Clebsch–Gordan 系数给出，并满足选择规则 \(|l_1-l_2|\le l_3\le l_1+l_2\) 与 \(l_3\le l_{\max}\)。若一个 interaction block 内包含多次耦合，则可表示为算子复合
   \[
   h^{(t)}=T^{(t)}_{m}\circ \cdots \circ T^{(t)}_{2}\circ T^{(t)}_{1}(h^{(t-1)}),
   \]
   其中每个 \(T^{(t)}_{k}\) 具有上述 intertwiner 结构。

2. **\(l=0\) 投影与 ACE 型不变量基（coupling graph）**：记 \(P^{(0)}:V^{(t)}\to (\mathbb{R}^{c_0^{(t)}}\otimes \mathcal{H}_0)\cong \mathbb{R}^{c_0^{(t)}}\) 为投影到标量不可约表示（\(l=0\)）的正交投影。则在给定耦合路径（即一组有限的耦合图/耦合树，coupling graph）\(\Gamma\) 与径向/角向截断索引集合 \(\Lambda\) 下，可将层内多次耦合与最终 \(l=0\) 投影写为
   \[
   \psi_{\Gamma,\Lambda}^{(t)}:=P^{(0)}\Big(T^{(t)}_{m}\circ \cdots \circ T^{(t)}_{1}(h^{(t-1)})\Big),
   \]
   其中 \((\Gamma,\Lambda)\) 唯一确定了耦合顺序、不可约表示耦合的中间角动量通道、以及径向基/元素通道的取值。由此得到的 \(\{\psi_{\Gamma,\Lambda}^{(t)}\}\) 构成在固定截断配置下的 ACE 型不变量基（或其有限维张成空间）；模型读出可视为在该有限维不变量空间上的可学习映射（线性组合与后续 MLP 等）。

3. **体阶与截断约束**：层内多次 TP 允许生成高体阶项，但在有限截断配置下，实际可生成的不变量集合受到如下约束：
   - 角动量截断 \(l_{\max}\)：限制了可生成的不可约表示类型
   - TP 路径集合：实际实现中仅允许部分 TP 路径组合
   - 通道数 \(\{c_l\}\)：限制了不变量子空间的维度

因此，MACE 在固定截断配置下生成的不变量子空间为
\[
\mathcal{V}_{\text{MACE}} = \text{span}\{\psi_{\text{MACE}}^{(t)} : t=1,\ldots,L\},
\]
其维度与体阶上界受上述截断参数约束。

### 5.2 跨层融合：跨深度双线性交叉不变量

跨层融合通过引入跨深度的双线性交叉不变量 \(\mathcal{I}(h^{(a)},h^{(b)})\) 扩展不变量特征集合，其中 \(h^{(a)},h^{(b)}\) 来自不同深度 \(a < b\)。

**构造机制**：跨层融合直接对不同深度的等变特征进行双线性不变量构造：
\[
\psi_{\text{cross}}^{(a,b)} = \mathcal{I}(h^{(a)},h^{(b)}),
\]
其中 \(\mathcal{I}(\cdot,\cdot)\) 为不变量双线性型（Definition 4），通过等变张量积后投影到 \(l=0\) 分量，或通过 Gram 型 contraction 实现。该构造与 MACE 的不变量读出在算子层面同属于“耦合后取标量分量/收缩”这一类操作；差异仅在于其自变量来自不同深度 \(a<b\) 的特征空间。

**体阶性质**：由 Lemma 2，若 \(\mathrm{bo}(h^{(a)})\le K_a\)、\(\mathrm{bo}(h^{(b)})\le K_b\)，则
\[
\mathrm{bo}(\psi_{\text{cross}}^{(a,b)})\le K_a+K_b.
\]
在 Theorem 2 的非退化条件下，该上界可达。

**不变量子空间扩展**：在相同 \(L,l_{\max},\{c_l\}\) 与路径截断配置下，跨层融合构造的不变量子空间为
\[
\mathcal{V}_{\text{extended}} = \mathcal{V}_{\text{MACE}} + \text{span}\{\psi_{\text{cross}}^{(a,b)} : 1\le a < b \le L\},
\]
其中 \(+\) 表示子空间的并（span）。若存在 \((a,b)\) 使得 \(\psi_{\text{cross}}^{(a,b)} \notin \mathcal{V}_{\text{MACE}}\)，则 \(\dim(\mathcal{V}_{\text{extended}}) > \dim(\mathcal{V}_{\text{MACE}})\)。

### 5.2.1 与 MACE 的严格对照（生成元集合与函数类包含）

为避免将差异归因于“是否能够构造多体”，以下在固定截断配置下以生成元集合、特征空间与最优逼近误差为对象给出可验证的结论。

设第 \(t\) 层等变特征在固定截断配置下取值于有限维表示空间 \(V_t\)。记同层二次不变量生成集合
\[
\mathcal{G}_{\mathrm{self}}(t):=\{\mathcal{I}(u,u): u\in V_t\},
\]
以及跨层二次不变量生成集合
\[
\mathcal{G}_{\mathrm{cross}}(a,b):=\{\mathcal{I}(u,v): u\in V_a,\, v\in V_b\},\qquad a<b.
\]

在给定的数据分布（或经验测度）诱导的内积 \(\langle \cdot,\cdot\rangle_{L^2}\) 下，设基线模型在读出端所使用的不变量特征空间为某个有限维线性子空间
\[
\mathcal{W}_{\mathrm{base}}\subset \mathrm{span}(\mathcal{G}_{\mathrm{self}}(L)).
\]
跨层融合在读出端加入跨层生成元后，对应的不变量特征空间定义为
\[
\mathcal{W}_{\mathrm{cross}}:=\mathcal{W}_{\mathrm{base}}+\mathrm{span}(\mathcal{G}_{\mathrm{cross}}(a,b)),
\]
其中 \(+\) 表示线性张成意义下的子空间和。

**Proposition 3（函数类包含）**：若读出网络类在特征输入维度扩展后保持不变且具有充分表达性，则由 \(\mathcal{W}_{\mathrm{base}}\) 扩展至 \(\mathcal{W}_{\mathrm{cross}}\) 诱导函数类包含关系。

**Proof.** 取读出网络参数使其忽略新增特征即可（Lemma 5）。□

**Proposition 4（最优 \(L^2\) 逼近误差的严格下降：充分条件）**：设目标函数 \(f\in L^2\)，并令
\[
e_{\mathrm{base}}:=\inf_{g\in \mathcal{W}_{\mathrm{base}}}\|f-g\|_{L^2},\qquad
e_{\mathrm{cross}}:=\inf_{g\in \mathcal{W}_{\mathrm{cross}}}\|f-g\|_{L^2}.
\]
则 \(e_{\mathrm{cross}}\le e_{\mathrm{base}}\)。进一步，若存在 \(w\in \mathrm{span}(\mathcal{G}_{\mathrm{cross}}(a,b))\) 使得其在 \(\mathcal{W}_{\mathrm{base}}\) 的正交补上的分量与 \(f\) 在该正交补上的分量具有非零相关（例如 \(\langle P_{\mathcal{W}_{\mathrm{base}}^\perp}f,\ P_{\mathcal{W}_{\mathrm{base}}^\perp}w\rangle_{L^2}\neq 0\)），则 \(e_{\mathrm{cross}}< e_{\mathrm{base}}\)。

**Proof.** 由 \(\mathcal{W}_{\mathrm{base}}\subseteq \mathcal{W}_{\mathrm{cross}}\) 立得 \(e_{\mathrm{cross}}\le e_{\mathrm{base}}\)。令 \(r:=P_{\mathcal{W}_{\mathrm{base}}^\perp}f\) 为基线残差。若存在 \(w\) 使得 \(P_{\mathcal{W}_{\mathrm{base}}^\perp}w\) 与 \(r\) 非正交，则沿该方向对 \(P_{\mathcal{W}_{\mathrm{base}}}f\) 作一次线性修正可严格降低残差范数，故最优误差严格下降。□

**Corollary 2（经验秩判据：可检验的严格扩展充分条件）**：在有限样本 \(\{x_n\}_{n=1}^N\) 上，令 \(\Phi_{\mathrm{base}}\in\mathbb{R}^{N\times d}\) 为基线不变量特征矩阵，\(\Phi_{\mathrm{cross}}\in\mathbb{R}^{N\times d'}\) 为加入跨层特征后的特征矩阵。若
\[
\mathrm{rank}(\Phi_{\mathrm{cross}})>\mathrm{rank}(\Phi_{\mathrm{base}}),
\]
则 \(\mathcal{W}_{\mathrm{cross}}\) 在经验内积意义下严格扩展 \(\mathcal{W}_{\mathrm{base}}\)；并且若目标 \(f\) 在样本上的残差向量与新增特征张成的子空间存在非零相关，则经验最小二乘误差严格下降。

### 5.3 关于优化性质的补充说明
跨层融合在结构上引入对中间层特征的额外读出路径，从而改变梯度传播路径。本文主要关注其在不变量生成子空间层面的结构性差异；关于具体优化动力学的定量分析不在本文讨论范围内。

---

## 6. 与实现的一致性说明（Implementation Mapping）

本文的抽象 \(\mathcal{I}(\cdot,\cdot)\) 覆盖下列实现：
- **e3nn**：`FullyConnectedTensorProduct` +（取 \(0e\)）与 `ElementwiseTensorProduct(...)->0e`
- **Cartesian（strict/loose）**：CG 等变张量积或其近似版本 + 0e 不变量
- **PureCartesian（dense/sparse）**：按 rank 的 δ/ε contraction 与 Gram 型不变量
- **ICTD-irreps**：按 \(l\) 的 \(m\) 指标 contraction / Gram 型不变量

---

## 7. 结论性表述

> **命题**：在固定消息传递深度与有限截断配置（例如 \(l_{\max}\)、通道数与张量积路径集合受限）下，跨层融合通过引入跨深度双线性交叉不变量 \(\mathcal{I}(h^{(a)},h^{(b)})\) 扩展不变量特征集合。该交叉项的体阶满足上界 \(K_a+K_b\)（Theorem 1），并在非退化条件下可达（Theorem 2）。上述结论刻画了跨层融合在不变量生成机制层面的结构差异。

---

## 8. Related Work：现有 MLIP 的跨深度机制分类

本节按数学结构将现有 MLIP 中常见的“跨层”机制划分为两类，并澄清本文方法的区别所在。

### 8.1 A 类：逐层读出累加（layer-wise additive readout）
该类方法在每个 interaction block（或 message passing layer）后施加读出，并将各层标量贡献累加：
\[
E_i=\sum_{t=1}^{L}\rho_t\!\left(\mathrm{Inv}(h_i^{(t)})\right),
\]
其中 \(\mathrm{Inv}(\cdot)\) 表示任意旋转不变的读出（例如对 \(l=0\) 分量的线性读出，或对等变特征的同层二次不变量），\(\rho_t\) 为标量网络。本文仅将该类结构作为抽象类型陈述，而不对具体文献实现逐一列举。

### 8.2 B 类：跨深度交叉不变量（cross-depth bilinear invariants，本文）
本文方法的核心不同在于显式构造跨深度的双线性交叉不变量：
\[
E_i=\rho\!\left(\ldots,\ \mathcal{I}(h_i^{(a)},h_i^{(b)}),\ \ldots\right),\qquad a<b,
\]
并将其与同层自项共同输入读出网络。由 Lemma 2 及 Theorem 1–2，可得该结构在固定深度与有限截断配置下引入额外的不变量生成元，其体阶上界为 \(K_a+K_b\)，并在非退化条件下可达。本文据此将其与逐层读出累加机制区分。

概述：A 类对应逐层读出累加；B 类对应跨深度双线性交叉不变量。两者在不变量构造方式上存在结构差异。

### 8.3 代表性 MLIP 对比（定性结构对照）
为便于对照，本节给出若干代表性 MLIP 的定性结构比较。下表不旨在穷尽各模型的全部实现变体；同一模型在不同代码库中可能存在“仅末层读出”与“逐层读出累加”等实现差异。本文仅对常见读出范式作结构性归类。

<table>
<thead>
<tr>
<th>模型/家族</th>
<th>主要对称性/表示</th>
<th>多体构造的主要来源</th>
<th>逐层读出累加（A 类）</th>
<th>跨深度交叉不变量（B 类）</th>
</tr>
</thead>
<tbody>
<tr>
<td>SchNet</td>
<td>SE(3) 平移不变、旋转不变（连续滤波卷积）</td>
<td>多层消息传递的非线性叠加</td>
<td>可选</td>
      <td rowspan="7">在实现中未显式引入</td>
</tr>
<tr>
<td>PhysNet</td>
<td>旋转不变（显式物理先验 + 消息传递）</td>
<td>逐层残差修正与消息传递</td>
<td>可选</td>
</tr>
<tr>
<td>DimeNet / DimeNet++</td>
<td>旋转不变（角向基/三体消息）</td>
<td>多个交互块 + 输出块（output blocks）</td>
<td>可选</td>
</tr>
<tr>
<td>GemNet（系列）</td>
<td>旋转不变（高阶几何消息）</td>
<td>多模块/多输出块</td>
<td>可选</td>
</tr>
<tr>
<td>PaiNN</td>
<td>SO(3) 等变（标量+向量通道）</td>
<td>等变消息传递与非线性</td>
<td>可选</td>
</tr>
<tr>
<td>NequIP / Allegro</td>
<td>E(3) 等变（irreps）</td>
<td>层内 TP + 多层消息传递</td>
<td>可选</td>
</tr>
<tr>
<td>MACE</td>
<td>E(3) 等变（ACE/TP 型多体基）</td>
<td>同层内进行多次张量积（TP）构造多体不变量；通过层内 TP 路径生成 ACE 风格的多体基</td>
<td>可选（依实现而异）</td>
</tr>
<tr>
<td>ACE/MTP（显式基）</td>
<td>旋转不变（显式多体不变量基）</td>
<td>显式 N-body 基函数截断</td>
<td>不适用</td>
<td>不适用</td>
</tr>
</tbody>
</table>

**对照要点（与 MACE 的差异表述建议）**：
1. MACE 的核心优势在于：在单个 interaction block 内通过张量积路径生成 ACE 风格的多体不变量（并在有限 \(l_{\max}\)、有限 multiplicity 与有限路径截断下形成有限维子空间）。
2. 跨层融合在不增加消息传递深度的前提下引入跨深度双线性交叉不变量 \(\mathcal{I}(h^{(a)},h^{(b)})\)，从而在固定截断配置下扩展可生成不变量子空间（参见 Lemma 2 / Theorem 2）。

### 8.4 对比结论

> 若干神经网络势能模型采用逐层读出并累加能量贡献（layer-wise additive readout）。该机制与本文所讨论的跨深度双线性交叉不变量不同：本文在等变表征 \(h^{(a)},h^{(b)}\) 之间引入 \(\mathcal{I}(h^{(a)},h^{(b)})\)，从而在固定深度与有限截断配置下扩展不变量生成集合，并在非退化条件下使交叉项的体阶上界 \(K_a+K_b\) 可达（Theorem 2）。
