# 分子力场模型架构分析

## 概述

本项目的所有模型实现都遵循**相同的架构框架**，唯一的区别在于**张量积（Tensor Product）的实现方式**。本文档详细分析了模型的统一框架和不同变体的实现差异。

> 论文级严格理论推导（跨层融合为何能显著超越 MACE）：见 `PAPER_CROSS_LAYER_FUSION_THEORY.md`。

---

## 统一架构框架

### 1. 整体流程

所有模型都遵循以下统一的前向传播流程：

```
输入 (pos, A, batch, edges) 
  ↓
【第一步卷积】E3Conv / CartesianE3Conv / PureCartesianE3Conv
  ↓ f1: (N, channels × irreps_dim)
【第二步卷积】E3Conv2 / CartesianE3Conv2 / PureCartesianE3Conv2  
  ↓ f2: (N, channels × irreps_dim)
【特征组合】concat([f1, f2])
  ↓ f_combine: (N, 2×channels × irreps_dim)
【Product 3】不变量提取 → 32个标量
  ↓ f_prod3: (N, 32)
【跨层融合】⚠️ 不同实现有不同策略
  ↓ 
  - e3nn: T = [f1, f2, f_combine_product]  (三层融合，保留原始特征)
  - Cartesian: T = [f_combine, f_prod3]  (两层融合，信息压缩)
  - PureCartesian: f_prod5 = [inv1, inv2, inv3]  (细粒度，独立提取)
  ↓ T: 不同维度（取决于实现）
【Product 5】元素级张量积 → 标量不变量
  ↓ f_prod5: (N, output_dim)
【Readout】MLP + WeightedSum
  ↓
输出：原子能量 (N, 1)
```

### 2. 核心组件

#### 2.1 输入处理

所有模型共享相同的输入格式：
- `pos`: 原子位置 (N, 3)
- `A`: 原子类型索引 (N,)
- `batch`: 批次索引 (N,)
- `edge_src`, `edge_dst`: 边的源和目标节点
- `edge_shifts`: 周期性边界条件偏移
- `cell`: 晶胞矩阵 (B, 3, 3)

#### 2.2 原子嵌入

```python
# 所有模型都使用相同的原子嵌入
atom_embedding = nn.Embedding(max_atomvalue, embedding_dim)
atom_mlp = MLP(embedding_dim → output_size)
Ai = atom_mlp(atom_embedding(A))  # (N, output_size)
```

#### 2.3 边几何信息

```python
# 计算相对位置向量（考虑PBC）
edge_vec = pos[edge_dst] - pos[edge_src] + shift_vecs
edge_length = edge_vec.norm(dim=1)
n = edge_vec / edge_length  # 归一化方向向量
```

#### 2.4 径向基函数

```python
# 使用e3nn的soft_one_hot_linspace
emb = soft_one_hot_linspace(
    edge_length, 0.0, max_radius, number_of_basis,
    basis=function_type, cutoff=True
)
```

---

## 第一步卷积（E3Conv / Conv1）

### 功能
将原子类型嵌入和边方向信息转换为高阶不可约表示（irreps）或秩张量。

### 统一流程

```
1. 原子嵌入: Ai = MLP(Embedding(A))  # (N, output_size)

2. 边方向特征构建:
   - e3nn: Y_l = spherical_harmonics(edge_vec, lmax)  # (E, 2l+1)
   - Cartesian: Y_l = CG_decomposition(edge_vec)       # (E, 2l+1)  
   - PureCartesian: n^{⊗L} = edge_rank_powers(n, Lmax) # (E, 3^L)
   - ICTD: Y_l = direction_harmonics(n, lmax)           # (E, 2l+1)

3. 初始特征构建:
   f_in = Ai[edge_src] ⊗ Y(edge_vec)
   # e3nn: tensor_product(Ai, Y_l)
   # PureCartesian: Ai * n^{⊗L} (外积)

4. 径向权重:
   weights = MLP(radial_basis(edge_length))  # (E, weight_numel)

5. 张量积（核心差异）:
   edge_features = TP(f_in, Ai[edge_dst], weights)
   # 不同实现使用不同的TP方法

6. 聚合到节点:
   out = scatter(edge_features, edge_dst) / neighbor_count
```

### 不同实现的差异

| 实现 | 边方向特征 | 张量积方法 | 输出维度 |
|------|-----------|-----------|---------|
| **e3nn** | 球谐函数 Y_l(n) | `o3.FullyConnectedTensorProduct` | `channels × sum(2l+1)` |
| **Cartesian** | CG分解 | `EquivariantTensorProduct` (CG系数) | `channels × sum(2l+1)` |
| **Cartesian-Loose** | CG分解 | `CartesianFullyConnectedTensorProduct` (范数乘积) | `channels × sum(2l+1)` |
| **PureCartesian** | n^{⊗L} (3^L) | `PureCartesianTensorProductO3` (δ/ε收缩) | `channels × sum(3^L)` |
| **PureCartesian-Sparse** | n^{⊗L} | `PureCartesianTensorProductO3Sparse` (限制路径) | `channels × sum(3^L)` |
| **ICTD-Irreps** | 调和多项式 | `HarmonicFullyConnectedTensorProduct` (ICTD) | `channels × (lmax+1)^2` |

---

## 第二步卷积（E3Conv2 / Conv2）

### 功能
进一步处理节点特征，使用边方向信息进行消息传递。

### 统一流程

```
1. 输入: f_in (N, channels × irreps_dim)

2. 边方向特征（同上）

3. 径向权重:
   weights = MLP(radial_basis(edge_length))

4. 张量积:
   edge_features = TP(f_in[edge_src], Y(edge_vec), weights)

5. 聚合:
   out = scatter(edge_features, edge_dst) / neighbor_count
```

### 关键差异
- **输入**: Conv1输出的是标量特征与边方向的组合，Conv2输入的是已经包含方向信息的节点特征
- **张量积**: 使用相同的TP实现，但输入特征不同

---

## 特征组合与不变量提取

### ⚠️ 重要发现：跨层融合机制

不同实现使用了**不同的跨层融合策略**，这是架构的关键差异：

#### 1. e3nn模式：三层融合（最复杂）

```python
# e3nn_layers.py
f_combine = torch.cat([f1, f2], dim=-1)
f_combine_product = self.product_3(f_combine, f_combine)  # 从组合特征提取不变量
T = torch.cat([f1, f2, f_combine_product], dim=-1)  # 三层融合！
```

**特点**:
- 保留了原始的 `f1` 和 `f2`（第一层和第二层卷积的独立输出）
- 同时融合了 `f_combine_product`（从f1+f2组合中提取的不变量）
- **跨层信息流**: f1 → f_combine → f_combine_product → T
- 信息最丰富，但维度最高

#### 2. Cartesian模式：两层融合

```python
# cartesian_e3_layers.py
f_combine = reorder_concatenated_irreps(f1, f2, ...)
f_prod3 = self.product_3(f_combine, f_combine)
T = torch.cat([f_combine, f_prod3], dim=-1)  # 两层融合
```

**特点**:
- 只保留组合后的 `f_combine`（f1和f2的组合）
- 融合 `f_prod3`（从f_combine提取的不变量）
- **信息流**: f1+f2 → f_combine → f_prod3 → T
- 维度中等，信息压缩

#### 3. PureCartesian模式：细粒度融合

```python
# pure_cartesian_layers.py
f_combine = merge_by_rank_o3(...)  # f1和f2的组合
f_prod3 = self.product_3(f_combine)  # 从组合特征提取32个标量
# 分别从f1和f2提取不变量
inv1 = self.product_5_o3(f1, f1)  # f1的自内积不变量
inv2 = self.product_5_o3(f2, f2)  # f2的自内积不变量
inv3 = f_prod3 * f_prod3  # f_prod3的平方
f_prod5 = torch.cat([inv1, inv2, inv3], dim=-1)  # 细粒度融合
```

**特点**:
- 分别从 `f1` 和 `f2` 提取独立的不变量
- 同时使用组合特征的不变量 `f_prod3`
- **信息流**: f1 → inv1, f2 → inv2, f_combine → f_prod3 → inv3
- 最细粒度的特征提取，保留各层的独立信息

#### 跨层融合对比表

| 模式 | T的组成 | 信息保留 | 维度 | 复杂度 |
|------|---------|---------|------|--------|
| **e3nn** | `[f1, f2, f_combine_product]` | ✅ 原始+组合+不变量 | 最高 | 最高 |
| **Cartesian** | `[f_combine, f_prod3]` | ⚠️ 组合+不变量 | 中等 | 中等 |
| **PureCartesian** | `[inv1, inv2, inv3]` | ✅ 独立不变量+组合不变量 | 最低 | 中等 |

---

### Product 3: 高阶不变量

**目标**: 从组合特征 `f_combine = concat([f1, f2])` 中提取标量不变量

#### e3nn实现
```python
# 使用FullyConnectedTensorProduct，输出限制为0e（标量）
product_3 = o3.FullyConnectedTensorProduct(
    irreps_in1=irreps_combine,
    irreps_in2=irreps_combine,
    irreps_out="32x0e",  # 只输出标量
    shared_weights=True,
    internal_weights=True
)
f_prod3 = product_3(f_combine, f_combine)  # (N, 32)
```

#### Cartesian实现
```python
# 使用EquivariantTensorProduct，通过CG系数耦合
product_3 = EquivariantTensorProduct(
    irreps_in1=irreps_combine,
    irreps_in2=irreps_combine,
    irreps_out="32x0e"
)
f_prod3 = product_3(f_combine, f_combine)  # (N, 32)
```

#### PureCartesian实现
```python
# 使用δ-收缩（内积）构建Gram矩阵
class PureCartesianInvariantBilinear:
    def forward(self, x):
        blocks = split_by_rank_o3(x, channels, Lmax)
        for L in range(Lmax + 1):
            t = blocks[L]  # (N, channels, 3^L)
            gram = einsum("nci,ndi->ncd", t, t) / sqrt(3^L)
            out += einsum("ocd,ncd->no", W[L], gram)
        return out  # (N, 32)
```

#### ICTD实现
```python
# 使用调和基的Gram矩阵
for l in range(lmax + 1):
    t = xb[l]  # (N, 2C, 2l+1)
    gram = einsum("ncm,ndm->ncd", t, t) / sqrt(2l+1)
    scalars += einsum("ocd,ncd->no", W_read[l], gram)
```

### Product 5: 元素级不变量

**目标**: 从 `T = concat([f1, f2, f_prod3])` 中提取最终不变量

#### e3nn实现
```python
# ElementwiseTensorProduct，每个irrep块内积
product_5 = o3.ElementwiseTensorProduct(
    irreps_in=T_irreps,
    irreps_out=["0e"],  # 只输出标量
    normalization='component'
)
f_prod5 = product_5(T, T)  # (N, output_dim)
```

#### PureCartesian实现
```python
# 对每个rank块计算自内积
class PureCartesianElementwiseTensorProductO3:
    def forward(self, x1, x2):
        blocks1 = split_by_rank_o3(x1, channels, Lmax)
        blocks2 = split_by_rank_o3(x2, channels, Lmax)
        for L in range(Lmax + 1):
            # 对每个rank L，计算内积并归一化
            inv = (blocks1[L] * blocks2[L]).sum(dim=-1) / sqrt(3^L)
            outs.append(inv)
        return concat(outs)  # (N, channels × (Lmax+1))
```

#### ICTD实现
```python
# 对每个l块计算内积
def _irreps_elementwise_tensor_product_0e(x1, x2):
    b1 = _split_irreps(x1, channels, lmax)
    b2 = _split_irreps(x2, channels, lmax)
    for l in range(lmax + 1):
        inv = (b1[l] * b2[l]).sum(dim=-1) / sqrt(2l+1)
        outs.append(inv)
    return concat(outs)
```

---

## 读取层（Readout）

### 统一实现

```python
# 所有模型都使用相同的读取层
proj_total = MainNet(
    input_dim=product_5_output_dim,
    hidden_sizes=[128, 128, 128],
    output_size=17
)
weighted_sum = RobustScalarWeightedSum(17, init_weights='zero')

# 前向传播
product_proj = proj_total(f_prod5)  # (N, 17)
e_out = weighted_sum(product_proj)  # (N, 17)
atom_energies = e_out.sum(dim=-1, keepdim=True)  # (N, 1)
```

---

## 张量积方法对比

### 1. e3nn (球谐函数 + CG系数)

**原理**: 
- 使用球谐函数 Y_l^m(n) 作为边方向基
- 使用Clebsch-Gordan系数进行不可约表示耦合
- 严格等变，数学上最标准

**优点**:
- 严格等变
- 维度效率高 (2l+1 vs 3^L)
- 成熟的实现

**缺点**:
- 需要计算球谐函数
- CG系数预计算或运行时计算开销

**代码位置**: `e3nn_layers.py`

---

### 2. Cartesian (CG分解 + 严格等变)

**原理**:
- 将笛卡尔向量分解为不可约表示
- 使用CG系数进行耦合（与e3nn数学等价）
- 严格等变

**优点**:
- 严格等变
- 避免球谐函数计算
- 与e3nn数学等价

**缺点**:
- 实现复杂度较高
- 性能可能略慢于e3nn

**代码位置**: `cartesian_e3_layers.py` (EquivariantTensorProduct)

---

### 3. Cartesian-Loose (范数乘积近似)

**原理**:
- 使用范数乘积 ||f1||^2 × ||f2||^2 作为标量近似
- 不严格等变，但旋转不变

**优点**:
- 速度快
- 支持torch.compile（推理时3-5x加速）
- 参数少

**缺点**:
- 不严格等变
- 可能损失方向信息

**代码位置**: `cartesian_e3_layers.py` (CartesianFullyConnectedTensorProduct)

---

### 4. PureCartesian (3^L 秩张量 + δ/ε收缩)

**原理**:
- 直接使用笛卡尔秩张量 (3^L维)
- 使用Kronecker delta (δ) 和Levi-Civita (ε) 进行收缩
- O(3)等变

**优点**:
- 最"纯"的笛卡尔实现
- 不需要球谐函数或CG系数
- 数学上直观

**缺点**:
- 维度高 (3^L vs 2l+1)
- 计算和存储开销大

**代码位置**: `pure_cartesian_layers.py`

---

### 5. PureCartesian-Sparse (限制路径的3^L)

**原理**:
- 与PureCartesian相同，但限制rank-rank交互路径
- 只允许特定的rank组合（如scalar-vector耦合）

**优点**:
- 减少计算量
- 保持O(3)等变性
- 比dense版本快

**缺点**:
- 可能损失表达能力
- 需要仔细设计路径策略

**代码位置**: `pure_cartesian_sparse_layers.py`

---

### 6. ICTD-Irreps (调和多项式 + trace-chain)

**原理**:
- 内部使用irreps表示（不是3^L）
- 边方向使用调和多项式（无需球谐函数）
- 张量积使用ICTD trace-chain投影

**优点**:
- 结合irreps效率和笛卡尔直观性
- 无需球谐函数
- 速度优化潜力

**缺点**:
- 实现复杂
- ICTD计算可能开销大

**代码位置**: `pure_cartesian_ictd_layers.py`

---

## 关键设计模式

### 1. 消息传递模式

所有模型都遵循标准的图神经网络消息传递：

```python
# 1. 计算边特征
edge_features = TP(node_features[src], edge_geometry, radial_weights)

# 2. 聚合到目标节点
node_features_new = scatter(edge_features, dst) / neighbor_count
```

### 2. 等变性保证

- **输入**: 原子位置 `pos` 是E(3)等变的
- **边方向**: `n = edge_vec / ||edge_vec||` 是旋转等变的
- **张量积**: 所有TP实现都保证输出等变性
- **不变量提取**: Product 3和Product 5确保最终输出是旋转不变的标量

### 3. 归一化策略

- **径向基函数**: `soft_one_hot_linspace` 带cutoff
- **边方向**: 归一化到单位向量
- **聚合**: 除以邻居数量（平均池化）
- **不变量**: 除以 `sqrt(2l+1)` 或 `sqrt(3^L)` (component normalization)

---

## 性能对比总结

| 方法 | 等变性 | 速度 | 内存 | 表达能力 | 推荐场景 |
|------|--------|------|------|----------|----------|
| **e3nn** | ✅ 严格 | 中等 | 低 | 高 | 标准训练 |
| **Cartesian** | ✅ 严格 | 中等 | 低 | 高 | 需要避免球谐函数 |
| **Cartesian-Loose** | ⚠️ 近似 | **快** | 低 | 中 | **推理加速** |
| **PureCartesian** | ✅ 严格 | 慢 | **高** | **最高** | 研究/分析 |
| **PureCartesian-Sparse** | ✅ 严格 | 中-快 | 中 | 高 | 平衡性能 |
| **ICTD-Irreps** | ✅ 严格 | 中 | 低 | 高 | 优化实验 |

---

## 代码结构对应关系

### 统一接口

所有Transformer层都实现相同的接口：

```python
class XxxTransformerLayer(nn.Module):
    def forward(self, pos, A, batch, edge_src, edge_dst, edge_shifts, cell):
        # 1. 第一步卷积
        f1 = self.e3_conv_emb(...)
        
        # 2. 第二步卷积  
        f2 = self.e3_conv_emb2(...)
        
        # 3. 特征组合
        f_combine = concat([f1, f2])
        
        # 4. Product 3
        f_prod3 = self.product_3(...)
        
        # 5. Product 5
        f_prod5 = self.product_5(...)
        
        # 6. Readout
        atom_energies = self.readout(f_prod5)
        
        return atom_energies
```

### 文件映射

| 模型类 | 文件 | Conv1 | Conv2 | Product3 | Product5 |
|--------|------|-------|-------|----------|----------|
| `E3_TransformerLayer_multi` | `e3nn_layers.py` | `E3Conv` | `E3Conv2` | `FullyConnectedTensorProduct` | `ElementwiseTensorProduct` |
| `CartesianTransformerLayer` | `cartesian_e3_layers.py` | `CartesianE3ConvSparse` | `CartesianE3Conv2Sparse` | `EquivariantTensorProduct` | `CartesianElementwiseTensorProduct` |
| `CartesianTransformerLayerLoose` | `cartesian_e3_layers.py` | `CartesianE3ConvSparseLoose` | `CartesianE3Conv2SparseLoose` | `CartesianFullyConnectedTensorProduct` | `CartesianElementwiseTensorProduct` |
| `PureCartesianTransformerLayer` | `pure_cartesian_layers.py` | `PureCartesianE3Conv` | `PureCartesianE3Conv2` | `PureCartesianInvariantBilinear` | `PureCartesianElementwiseTensorProductO3` |
| `PureCartesianSparseTransformerLayer` | `pure_cartesian_sparse_layers.py` | `PureCartesianSparseE3Conv` | `PureCartesianSparseE3Conv2` | `PureCartesianSparseInvariantBilinear` | `PureCartesianElementwiseTensorProductO3` |
| `PureCartesianICTDTransformerLayer` | `pure_cartesian_ictd_layers.py` | `ICTDIrrepsE3Conv` | `HarmonicFullyConnectedTensorProduct` | Gram矩阵 | `_irreps_elementwise_tensor_product_0e` |

---

## 总结

1. **统一架构**: 所有模型遵循相同的6步流程（Conv1 → Conv2 → Combine → Product3 → **跨层融合** → Product5 → Readout）

2. **核心差异1 - 张量积方法**: 仅在于张量积的实现方式：
   - 球谐函数 vs 笛卡尔分解 vs 秩张量
   - CG系数 vs δ/ε收缩 vs ICTD投影
   - 严格等变 vs 近似等变

3. **核心差异2 - 跨层融合策略** ⚠️:
   - **e3nn**: 三层融合 `T = [f1, f2, f_combine_product]` - 保留最完整信息
   - **Cartesian**: 两层融合 `T = [f_combine, f_prod3]` - 信息压缩
   - **PureCartesian**: 细粒度融合 `f_prod5 = [inv1, inv2, inv3]` - 独立提取各层不变量

3. **设计哲学**: 
   - 保持接口一致性，便于切换和对比
   - 在等变性、速度和表达能力之间权衡
   - 模块化设计，每个组件可独立替换

4. **实际应用**:
   - **训练**: 推荐使用严格等变版本（e3nn, Cartesian, PureCartesian）
   - **推理**: 可使用Loose版本加速（Cartesian-Loose + torch.compile）
   - **研究**: PureCartesian提供最直观的数学理解

