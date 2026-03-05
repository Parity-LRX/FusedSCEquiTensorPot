#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include <torch/script.h>
#include <torch/torch.h>

namespace mfftorch {

struct MFFOutputs {
  double energy = 0.0;
  torch::Tensor atom_energy;   // (ntotal,1) or (ntotal,) on engine device
  torch::Tensor forces;        // (ntotal,3) on engine device
  torch::Tensor atom_virial;   // (ntotal,6) on engine device — Voigt: xx,yy,zz,xy,xz,yz
};

class MFFTorchEngine {
 public:
  MFFTorchEngine() = default;

  void load_core(const std::string& core_pt_path, const std::string& device_str);

  // Warmup: run one dummy forward+backward to trigger JIT compilation and CUDA caching.
  void warmup(int64_t N = 32, int64_t E = 256);

  const torch::Device& device() const { return device_; }
  bool is_cuda() const { return device_.is_cuda(); }

  MFFOutputs compute(int64_t nlocal, int64_t ntotal,
                     const torch::Tensor& A,
                     const torch::Tensor& edge_src,
                     const torch::Tensor& edge_dst,
                     const torch::Tensor& rij,
                     bool need_energy = true,
                     bool need_atom_virial = false);

 private:
  torch::jit::script::Module core_;
  bool loaded_ = false;

  torch::Device device_{torch::kCPU};

  // Cached constant tensors.
  torch::Tensor cell_;  // (1,3,3) float32 on device

  // Reusable per-step buffers (avoid repeated CUDA malloc).
  int64_t cached_ntotal_ = 0;
  int64_t cached_nedges_ = 0;
  torch::Tensor buf_pos_;
  torch::Tensor buf_batch_;
  torch::Tensor buf_edge_shifts_;
};

}  // namespace mfftorch
