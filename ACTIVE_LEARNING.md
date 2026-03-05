# 主动学习 (mff-active-learn)

主动学习实现 DPGen2 风格的工作流：**训练集成 → 探索（MD/NEB）→ 按力偏差筛选 → DFT 标注 → 合并数据 → 重复**，用于在势能面尚未覆盖的区域自动采点并扩充训练集。

支持 **单节点**（本机多进程并发标注）和 **超算**（SLURM 按结构提交作业）。同一套 DFT 脚本模板可同时用于本地测试（`local-script`）和集群（`slurm`）。

---

## 快速开始示例

### PySCF（本地，无需外部 DFT 二进制）

```bash
mff-active-learn --explore-type ase --explore-mode md --label-type pyscf \
    --pyscf-method b3lyp --pyscf-basis 6-31g* \
    --label-n-workers 8 --md-steps 500 --n-iterations 5
```

### VASP（单节点，ASE 接口）

```bash
export ASE_VASP_COMMAND="mpirun -np 4 vasp_std"
export VASP_PP_PATH=/path/to/potpaw_PBE

mff-active-learn --explore-type ase --label-type vasp \
    --vasp-xc PBE --vasp-encut 500 --vasp-kpts 4 4 4 \
    --label-n-workers 8 --label-threads-per-worker 4 \
    --md-steps 1000 --n-iterations 10
```

### SLURM（超算，每结构一个作业）

```bash
mff-active-learn --explore-type ase --label-type slurm \
    --slurm-template dft_job.sh --slurm-partition cpu \
    --slurm-nodes 1 --slurm-ntasks 32 --slurm-time 04:00:00
```

### local-script（本地执行脚本模板）

```bash
mff-active-learn --explore-type ase --label-type local-script \
    --local-script-template dft_job.sh \
    --label-n-workers 4 \
    --md-steps 500 --n-iterations 3
```

---

## 核心参数表

| 参数 | 默认 | 说明 |
|------|------|------|
| `--work-dir` | `al_work` | 主动学习工作目录 |
| `--data-dir` | `data` | 训练数据目录（含 processed_train.h5 或 train.xyz） |
| `--init-structure` | 自动从 data-dir 提取 | 初始结构 XYZ 路径 |
| `--n-models` | 4 | 集成模型数量 |
| `--n-iterations` | 20 | 单阶段最大迭代次数（不用 stages 时） |
| `--explore-type` | 必选 | 探索后端：`ase` 或 `lammps` |
| `--explore-mode` | `md` | 探索方式：`md`（分子动力学）或 `neb`（弹性带） |
| `--label-type` | 必选 | 标注方式，见下表 |
| `--md-temperature` | 300 | MD 温度 (K) |
| `--md-steps` | 10000 | MD 步数 |
| `--md-timestep` | 1.0 | MD 时间步长 (fs) |
| `--md-friction` | 0.01 | Langevin 摩擦系数 |
| `--md-relax-fmax` | 0.05 | 预优化力收敛阈值 (eV/Å) |
| `--md-log-interval` | 10 | 轨迹记录间隔 |
| `--level-f-lo` / `--level-f-hi` | 0.05 / 0.5 | 力偏差筛选阈值 (eV/Å) |
| `--conv-accuracy` | 0.9 | 收敛判定比例 |
| `--epochs` | 由 mff-train 默认 | 每轮每个模型的训练 epoch 数 |
| `--stages` | 无 | 多阶段 JSON 文件路径 |
| `--device` | `cuda` | 推理设备 |
| `--max-radius` | 5.0 | 邻居搜索最大半径 (Å) |
| `--atomic-energy-file` | `data/fitted_E0.csv` | 原子参考能量 CSV |
| `--neb-initial` / `--neb-final` | 无 | NEB 模式下的初/末结构 |

---

## 标注类型 (--label-type)

| 类型 | 说明 | 典型用途 |
|------|------|----------|
| `identity` | 用当前 ML 模型预测（不跑 DFT） | 调试、快速测试流程 |
| `pyscf` | PySCF 计算（无需外部二进制） | 小分子、本地验证 |
| `vasp` | VASP（ASE 接口） | 平面波 DFT，单节点或脚本内 MPI |
| `cp2k` | CP2K（ASE 接口） | 高斯+平面波，单节点 |
| `espresso` | Quantum Espresso pw.x（ASE 接口） | 单节点 QE |
| `gaussian` | Gaussian g16/g09（ASE 接口） | 单节点 |
| `orca` | ORCA（ASE 接口） | 单节点 |
| `script` | 用户脚本：`脚本路径 input.xyz output.xyz` | 任意 DFT/程序 |
| `local-script` | 与 SLURM 同格式的脚本模板，**本地执行** | 单节点 + 同一套脚本 |
| `slurm` | 与 local-script 同格式的脚本模板，**每结构提交一个 sbatch 作业** | 超算多节点 |

---

## 并发与线程控制

| 参数 | 默认 | 说明 |
|------|------|------|
| `--label-n-workers` | 1 | 同时跑多少个结构（进程数） |
| `--label-threads-per-worker` | 1（n_workers>1 时）或自动 | 每个结构内部线程数（如 PySCF/VASP 的 OpenMP） |
| `--label-error-handling` | `raise` | `raise`（任一失败即退出）或 `skip`（跳过失败结构继续） |

**建议**：`n_workers × threads_per_worker ≤ 总核数`，避免过载。

---

## 多阶段 JSON 格式 (--stages)

通过 JSON 定义多个阶段，每阶段可设不同温度、步数、迭代上限和收敛阈值。例如先 300K 再 600K：

```bash
mff-active-learn --explore-type ase --label-type pyscf --stages stages.json
```

`stages.json` 示例（数组，每项一个阶段）：

```json
[
  {
    "name": "300K",
    "temperature": 300,
    "nsteps": 500,
    "timestep": 1.0,
    "log_interval": 10,
    "level_f_lo": 0.05,
    "level_f_hi": 0.5,
    "conv_accuracy": 0.9,
    "max_iters": 5
  },
  {
    "name": "600K",
    "temperature": 600,
    "nsteps": 1000,
    "timestep": 1.0,
    "level_f_lo": 0.05,
    "level_f_hi": 0.5,
    "conv_accuracy": 0.9,
    "max_iters": 5
  }
]
```

也支持 `{"stages": [...]}` 包裹格式。未给出的字段会使用默认值。使用 `--stages` 时，命令行中的 `--md-*`、`--n-iterations` 等单阶段参数会被忽略。

---

## 脚本模板占位符

`local-script` 与 `slurm` 共用同一套占位符：

| 占位符 | 说明 |
|--------|------|
| `{run_dir}` | 当前结构的运行目录 |
| `{input_xyz}` | 输入 XYZ 路径 |
| `{output_xyz}` | 输出 extended XYZ 路径 |
| `{job_name}` | 作业/任务名称 |
| `{partition}` | SLURM 分区（仅 slurm） |
| `{nodes}` | 节点数（仅 slurm） |
| `{ntasks}` | 任务数（仅 slurm） |
| `{time}` | 时间限制（仅 slurm） |
| `{mem}` | 内存（仅 slurm） |

模板内其他 `{key}` 或 shell 变量（如 `$HOME`）会原样保留。

---

## SLURM 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--slurm-template` | 必选 | 作业脚本模板路径 |
| `--slurm-partition` | `cpu` | 队列/分区名 |
| `--slurm-nodes` | 1 | 每作业节点数 |
| `--slurm-ntasks` | 32 | 每作业任务数 |
| `--slurm-time` | `02:00:00` | 墙钟时间限制 |
| `--slurm-mem` | `64G` | 每作业内存 |
| `--slurm-max-concurrent` | 200 | 队列中最大并发作业数 |
| `--slurm-poll-interval` | 30 | 轮询 squeue 间隔（秒） |
| `--slurm-extra` | 无 | 额外 sbatch 参数，如 `--account=myproject` |
| `--slurm-cleanup` | False | 成功后删除运行目录 |

若某结构的 `output.xyz` 已存在（例如上次中断后重跑），会自动跳过该结构的提交（resume）。

---

## DFT 后端 CLI 参数

### PySCF

| 参数 | 默认 | 说明 |
|------|------|------|
| `--pyscf-method` | `b3lyp` | 方法：b3lyp, pbe, hf, mp2 等 |
| `--pyscf-basis` | `sto-3g` | 基组：sto-3g, 6-31g*, def2-svp 等 |
| `--pyscf-charge` | 0 | 总电荷 |
| `--pyscf-spin` | 0 | 2S（未配对电子数） |
| `--pyscf-max-memory` | 4000 | 最大内存 (MB) |
| `--pyscf-conv-tol` | 1e-9 | SCF 收敛阈值 |

### VASP

| 参数 | 默认 | 说明 |
|------|------|------|
| `--vasp-xc` | `PBE` | XC 泛函：PBE, LDA, HSE06 等 |
| `--vasp-encut` | 无 | 平面波截断 (eV) |
| `--vasp-kpts` | `1 1 1` | k 点网格 |
| `--vasp-ediff` | 1e-6 | SCF 收敛阈值 (eV) |
| `--vasp-ismear` | 0 | 展宽类型：0=Gaussian, -5=tetrahedron |
| `--vasp-sigma` | 0.05 | 展宽宽度 (eV) |
| `--vasp-command` | 覆盖 ASE_VASP_COMMAND | 运行命令 |
| `--vasp-cleanup` | False | 成功后删除运行目录 |

### CP2K

| 参数 | 默认 | 说明 |
|------|------|------|
| `--cp2k-xc` | `PBE` | XC 泛函 |
| `--cp2k-basis-set` | `DZVP-MOLOPT-SR-GTH` | 高斯基组 |
| `--cp2k-pseudo` | `auto` | 赝势名称 |
| `--cp2k-cutoff` | 400.0 | 平面波截断 (Ry) |
| `--cp2k-max-scf` | 50 | 最大 SCF 迭代 |
| `--cp2k-charge` | 0.0 | 总电荷 |
| `--cp2k-command` | 覆盖 ASE_CP2K_COMMAND | 运行命令 |
| `--cp2k-cleanup` | False | 成功后删除运行目录 |

### Quantum Espresso

| 参数 | 默认 | 说明 |
|------|------|------|
| `--qe-pseudo-dir` | 必选 | 赝势 .UPF 目录 |
| `--qe-pseudopotentials` | 必选 | JSON：`'{"H":"H.pbe.UPF","O":"O.pbe.UPF"}'` |
| `--qe-ecutwfc` | 60.0 | 波函数截断 (Ry) |
| `--qe-ecutrho` | 4*ecutwfc | 电荷密度截断 (Ry) |
| `--qe-kpts` | `1 1 1` | k 点网格 |
| `--qe-command` | 无 | pw.x 命令 |
| `--qe-cleanup` | False | 成功后删除运行目录 |

### Gaussian

| 参数 | 默认 | 说明 |
|------|------|------|
| `--gaussian-method` | `b3lyp` | 理论级别 |
| `--gaussian-basis` | `6-31+G*` | 基组 |
| `--gaussian-charge` | 0 | 总电荷 |
| `--gaussian-mult` | 1 | 自旋多重度 2S+1 |
| `--gaussian-nproc` | 1 | %nprocshared |
| `--gaussian-mem` | `4GB` | 内存 |
| `--gaussian-command` | 无 | 覆盖 Gaussian 命令 |
| `--gaussian-cleanup` | False | 成功后删除运行目录 |

### ORCA

| 参数 | 默认 | 说明 |
|------|------|------|
| `--orca-simpleinput` | `B3LYP def2-TZVP TightSCF` | 简单输入行（! 之后） |
| `--orca-nproc` | 1 | %pal nprocs |
| `--orca-charge` | 0 | 总电荷 |
| `--orca-mult` | 1 | 自旋多重度 2S+1 |
| `--orca-command` | 无 | ORCA 可执行路径 |
| `--orca-cleanup` | False | 成功后删除运行目录 |

---

## 环境变量

| 变量 | 说明 |
|------|------|
| `ASE_VASP_COMMAND` | VASP 运行命令，如 `mpirun -np 4 vasp_std` |
| `VASP_PP_PATH` | VASP 赝势目录 |
| `ASE_CP2K_COMMAND` | CP2K 运行命令，如 `cp2k_shell.psmp` |
| `CP2K_DATA_DIR` | CP2K 数据目录 |
| `ASE_GAUSSIAN_COMMAND` | Gaussian 命令，如 `g16 < PREFIX.com > PREFIX.log` |

ORCA 和 QE 可通过 `--orca-command`、`--qe-command` 指定，或确保可执行文件在 PATH 中。

---

## 使用示例汇总

**快速测试（不跑 DFT，用 ML 自举）：**

```bash
mff-active-learn --explore-type ase --explore-mode md --label-type identity \
    --md-temperature 300 --md-steps 200 --n-iterations 2 --epochs 5 \
    --n-models 2
```

**本地 PySCF + 多 worker：**

```bash
mff-active-learn --explore-type ase --label-type pyscf \
    --pyscf-method b3lyp --pyscf-basis 6-31g* \
    --label-n-workers 4 --md-steps 500 --n-iterations 5 --epochs 100
```

**超算 SLURM + 脚本模板：**

```bash
mff-active-learn --explore-type ase --label-type slurm \
    --slurm-template vasp_job.sh \
    --slurm-partition normal --slurm-ntasks 64 --slurm-time 08:00:00 \
    --stages stages.json --label-error-handling skip
```

**NEB 探索：**

```bash
mff-active-learn --explore-type ase --explore-mode neb \
    --neb-initial reactant.xyz --neb-final product.xyz \
    --label-type pyscf --pyscf-method b3lyp --n-iterations 5
```

---

## FAQ

### Q: 初始结构从哪里来？

若未指定 `--init-structure`，会从 `--data-dir` 的 `train.xyz` 或 `processed_train.h5` 取第一个结构。

### Q: 如何从上次中断处继续？

SLURM 模式下，若某结构的 `output.xyz` 已存在，会自动跳过该结构的提交。本地模式下，需手动处理或使用 `--label-error-handling skip` 跳过失败结构。

### Q: 输出 extended XYZ 格式要求？

标注器输出的 XYZ 需包含 `Properties=species:S:1:pos:R:3:energy:R:1:forces:R:3` 及相应数据，能量单位 eV，力单位 eV/Å。

### Q: 如何查看所有参数？

```bash
mff-active-learn --help
```

### Q: 训练超参数如何传递？

`--epochs` 会传给内部 `mff-train`。其他训练参数可通过扩展 `train_args` 传入（当前 CLI 仅暴露 `--epochs`）。
