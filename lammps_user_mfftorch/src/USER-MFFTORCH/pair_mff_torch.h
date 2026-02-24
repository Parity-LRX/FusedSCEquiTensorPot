#ifdef PAIR_CLASS
// clang-format off
PairStyle(mff/torch,PairMFFTorch);
// clang-format on
#else

#ifndef LMP_PAIR_MFF_TORCH_H
#define LMP_PAIR_MFF_TORCH_H

#include "pair.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mfftorch {
class MFFTorchEngine;
}

namespace LAMMPS_NS {

class PairMFFTorch : public Pair {
 public:
  PairMFFTorch(class LAMMPS *lmp);
  ~PairMFFTorch() override;

  void compute(int eflag, int vflag) override;
  void settings(int narg, char **arg) override;
  void coeff(int narg, char **arg) override;
  double init_one(int i, int j) override;
  void init_style() override;

 protected:
  void allocate();

  double cut_global_ = 0.0;
  double cutsq_global_ = 0.0;

  std::string device_str_ = "cuda";
  std::string core_pt_path_;

  std::vector<int64_t> type2Z_;

  std::unique_ptr<mfftorch::MFFTorchEngine> engine_;
  bool engine_loaded_ = false;
};

}  // namespace LAMMPS_NS

#endif  // LMP_PAIR_MFF_TORCH_H
#endif  // PAIR_CLASS
