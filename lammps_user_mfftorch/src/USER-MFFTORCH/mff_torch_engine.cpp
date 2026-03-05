#include "mff_torch_engine.h"

#include <c10/cuda/CUDAGuard.h>
#include <cstdlib>
#include <dlfcn.h>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mfftorch {

static int detect_local_gpu_index() {
  // Try common MPI environment variables for local rank.
  const char* env_vars[] = {
    "OMPI_COMM_WORLD_LOCAL_RANK",  // OpenMPI
    "MV2_COMM_WORLD_LOCAL_RANK",   // MVAPICH2
    "MPI_LOCALRANKID",             // Intel MPI
    "SLURM_LOCALID",               // SLURM
    "LOCAL_RANK",                   // PyTorch convention
    nullptr
  };
  for (const char** v = env_vars; *v; ++v) {
    const char* val = std::getenv(*v);
    if (val && val[0] != '\0') {
      int rank = std::atoi(val);
      int n_gpus = static_cast<int>(torch::cuda::device_count());
      if (n_gpus > 0) return rank % n_gpus;
    }
  }
  return 0;
}

static torch::Device pick_device(const std::string& device_str) {
  if (device_str.rfind("cuda:", 0) == 0) {
    int idx = std::atoi(device_str.c_str() + 5);
    if (!torch::cuda::is_available()) {
      throw std::runtime_error("requested " + device_str + " but CUDA is not available");
    }
    c10::cuda::set_device(idx);
    return torch::Device(torch::kCUDA, idx);
  }
  if (device_str == "cuda") {
    if (!torch::cuda::is_available()) {
      throw std::runtime_error("requested device=cuda but torch::cuda::is_available() is false");
    }
    int idx = detect_local_gpu_index();
    c10::cuda::set_device(idx);
    return torch::Device(torch::kCUDA, idx);
  }
  return torch::Device(torch::kCPU);
}

static void ensure_libpython() {
  // CPython extension modules expect Python C API symbols (e.g. PyExc_ValueError)
  // to be provided by the loading process. In a pure-C++ process like LAMMPS,
  // we must dlopen libpython first with RTLD_GLOBAL so those symbols are available.
  static bool done = false;
  if (done) return;
  done = true;

  const char* env = std::getenv("MFF_LIBPYTHON");
  if (env && env[0] != '\0') {
    if (dlopen(env, RTLD_LAZY | RTLD_GLOBAL)) return;
  }
  // Auto-detect: try common libpython names (relies on LD_LIBRARY_PATH).
  const char* names[] = {
    "libpython3.12.so", "libpython3.11.so", "libpython3.10.so",
    "libpython3.12.so.1.0", "libpython3.11.so.1.0", "libpython3.10.so.1.0",
    "libpython3.so",
    nullptr
  };
  for (const char** n = names; *n; ++n) {
    if (dlopen(*n, RTLD_LAZY | RTLD_GLOBAL)) return;
  }
}

static void load_custom_op_libs() {
  const char* env = std::getenv("MFF_CUSTOM_OPS_LIB");
  if (!env || env[0] == '\0') return;

  ensure_libpython();

  std::string paths(env);
  std::string::size_type start = 0;
  while (start < paths.size()) {
    auto pos = paths.find(':', start);
    std::string lib = (pos == std::string::npos)
                          ? paths.substr(start)
                          : paths.substr(start, pos - start);
    start = (pos == std::string::npos) ? paths.size() : pos + 1;
    if (lib.empty()) continue;
    void* handle = dlopen(lib.c_str(), RTLD_LAZY | RTLD_GLOBAL);
    if (!handle) {
      throw std::runtime_error(
          std::string("Failed to load custom ops library '") + lib + "': " + dlerror() +
          "\nSet MFF_CUSTOM_OPS_LIB to the path of cuequivariance ops .so"
          "\nIf 'undefined symbol: Py*', also set MFF_LIBPYTHON=/path/to/libpython3.XX.so");
    }
  }
}

void MFFTorchEngine::load_core(const std::string& core_pt_path, const std::string& device_str) {
  device_ = pick_device(device_str);
  load_custom_op_libs();
  core_ = torch::jit::load(core_pt_path, device_);
  core_.eval();

  try {
    core_ = torch::jit::freeze(core_);
  } catch (...) {
    // freeze may fail for some models; proceed without it.
  }

  loaded_ = true;
  cached_ntotal_ = 0;
  cached_nedges_ = 0;

  cell_ = torch::eye(3, torch::TensorOptions().dtype(torch::kFloat32).device(device_)).unsqueeze(0) * 100.0f;
}

void MFFTorchEngine::warmup(int64_t N, int64_t E) {
  if (!loaded_) return;

  auto A = torch::ones({N}, torch::TensorOptions().dtype(torch::kInt64).device(device_));
  auto edge_src = torch::zeros({E}, torch::TensorOptions().dtype(torch::kInt64).device(device_));
  auto edge_dst = torch::zeros({E}, torch::TensorOptions().dtype(torch::kInt64).device(device_));
  auto rij = torch::zeros({E, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));

  for (int i = 0; i < 3; i++) {
    compute(N, N, A, edge_src, edge_dst, rij, false);
  }
  if (device_.is_cuda()) torch::cuda::synchronize();
}

MFFOutputs MFFTorchEngine::compute(int64_t nlocal, int64_t ntotal,
                                  const torch::Tensor& A_in,
                                  const torch::Tensor& edge_src_in,
                                  const torch::Tensor& edge_dst_in,
                                  const torch::Tensor& rij_in,
                                  bool need_energy,
                                  bool need_atom_virial) {
  if (!loaded_) throw std::runtime_error("MFFTorchEngine not loaded");
  if (nlocal <= 0 || ntotal <= 0) return {};

  const int64_t nedges = rij_in.size(0);

  auto A = (A_in.device() == device_ && A_in.dtype() == torch::kInt64)
               ? A_in : A_in.to(device_, torch::kInt64);
  auto edge_src = (edge_src_in.device() == device_ && edge_src_in.dtype() == torch::kInt64)
                      ? edge_src_in : edge_src_in.to(device_, torch::kInt64);
  auto edge_dst = (edge_dst_in.device() == device_ && edge_dst_in.dtype() == torch::kInt64)
                      ? edge_dst_in : edge_dst_in.to(device_, torch::kInt64);
  auto rij = (rij_in.device() == device_ && rij_in.dtype() == torch::kFloat32)
                 ? rij_in : rij_in.to(device_, torch::kFloat32);

  if (cached_ntotal_ != ntotal) {
    buf_pos_ = torch::zeros({ntotal, 3},
                            torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    buf_batch_ = torch::zeros({ntotal},
                              torch::TensorOptions().dtype(torch::kInt64).device(device_));
    cached_ntotal_ = ntotal;
  }
  if (cached_nedges_ != nedges) {
    buf_edge_shifts_ = torch::zeros({nedges, 3},
                                    torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    cached_nedges_ = nedges;
  }

  auto pos = buf_pos_.detach().zero_().requires_grad_(true);

  // When per-atom virial is requested, also differentiate w.r.t. edge vectors.
  // Clone+detach to get an independent leaf tensor (separate storage from buf_rij_).
  torch::Tensor rij_leaf;
  if (need_atom_virial) {
    rij_leaf = rij.clone().detach().requires_grad_(true);
  } else {
    rij_leaf = rij;
  }

  auto edge_vec = pos.index_select(0, edge_dst) - pos.index_select(0, edge_src) + rij_leaf;

  std::vector<torch::jit::IValue> inputs;
  inputs.reserve(8);
  inputs.push_back(pos);
  inputs.push_back(A);
  inputs.push_back(buf_batch_);
  inputs.push_back(edge_src);
  inputs.push_back(edge_dst);
  inputs.push_back(buf_edge_shifts_);
  inputs.push_back(cell_);
  inputs.push_back(edge_vec);

  auto atom_e = core_.forward(inputs).toTensor();
  auto atom_e_flat = atom_e.view({atom_e.size(0)});
  auto E_local = atom_e_flat.narrow(0, 0, nlocal).sum();

  // Differentiate w.r.t. pos (forces) and optionally rij_leaf (edge forces).
  std::vector<torch::Tensor> grad_inputs = {pos};
  if (need_atom_virial) grad_inputs.push_back(rij_leaf);

  auto grads = torch::autograd::grad({E_local}, grad_inputs, {}, /*retain_graph=*/false,
                                     /*create_graph=*/false, /*allow_unused=*/true);
  auto forces = -grads[0];

  MFFOutputs out;
  out.atom_energy = atom_e;
  out.forces = forces;

  if (need_atom_virial && grads.size() > 1 && grads[1].defined()) {
    auto edge_forces = -grads[1];  // [nedges, 3]

    // Per-edge virial in Voigt order (LAMMPS convention):
    //   v = del ⊗ f_on_src,  where del = x[src]-x[dst] = -rij,  f_on_src = -edge_forces
    //   v = (-rij) ⊗ (-edge_forces) = rij ⊗ edge_forces
    auto r0 = rij_leaf.select(1, 0);  // [E]
    auto r1 = rij_leaf.select(1, 1);
    auto r2 = rij_leaf.select(1, 2);
    auto f0 = edge_forces.select(1, 0);
    auto f1 = edge_forces.select(1, 1);
    auto f2 = edge_forces.select(1, 2);

    auto edge_vir = torch::stack({
        r0 * f0,  // xx
        r1 * f1,  // yy
        r2 * f2,  // zz
        r0 * f1,  // xy
        r0 * f2,  // xz
        r1 * f2,  // yz
    }, 1);  // [nedges, 6]

    // Scatter half to each endpoint (gauge-invariant split).
    auto atom_vir = torch::zeros({ntotal, 6}, edge_vir.options());
    auto half_vir = 0.5f * edge_vir;
    auto src_idx = edge_src.unsqueeze(1).expand_as(half_vir);
    auto dst_idx = edge_dst.unsqueeze(1).expand_as(half_vir);
    atom_vir.scatter_add_(0, src_idx, half_vir);
    atom_vir.scatter_add_(0, dst_idx, half_vir);

    out.atom_virial = atom_vir;
  }

  if (need_energy) {
    out.energy = E_local.detach().to(torch::kCPU).item<double>();
  }
  return out;
}

}  // namespace mfftorch
