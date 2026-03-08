#include "pair_mff_torch.h"

#include "atom.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "input.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "update.h"
#include "utils.h"
#include "variable.h"

#include "mff_periodic_table.h"
#include "mff_torch_engine.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

using namespace LAMMPS_NS;

namespace {

std::string normalize_variable_name(const std::string &name) {
  if (name.rfind("v_", 0) == 0) return name.substr(2);
  return name;
}

}  // namespace

PairMFFTorch::PairMFFTorch(LAMMPS *lmp) : Pair(lmp) {
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
}

PairMFFTorch::~PairMFFTorch() {
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

void PairMFFTorch::allocate() {
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  for (int i = 1; i <= n; i++) {
    for (int j = 1; j <= n; j++) {
      setflag[i][j] = 0;
      cutsq[i][j] = 0.0;
    }
  }
}

void PairMFFTorch::settings(int narg, char **arg) {
  if (narg < 1) error->all(FLERR, "Illegal pair_style mff/torch command");
  cut_global_ = utils::numeric(FLERR, arg[0], false, lmp);
  if (cut_global_ <= 0.0) error->all(FLERR, "pair_style mff/torch cutoff must be > 0");
  cutsq_global_ = cut_global_ * cut_global_;

  use_external_field_ = false;
  external_tensor_rank_ = 0;
  external_field_symmetric_rank2_ = false;
  external_field_var_names_.clear();
  cached_external_field_values_.clear();
  external_tensor_cache_ = torch::Tensor();

  for (int i = 1; i < narg; ++i) {
    const std::string opt(arg[i]);
    if (opt == "cpu" || opt == "cuda") {
      device_str_ = opt;
      continue;
    }
    if (opt == "field") {
      if (i + 3 >= narg) {
        error->all(FLERR, "pair_style mff/torch field expects three equal-style variables: v_Ex v_Ey v_Ez");
      }
      use_external_field_ = true;
      external_tensor_rank_ = 1;
      external_field_symmetric_rank2_ = false;
      external_field_var_names_ = {
          normalize_variable_name(arg[i + 1]),
          normalize_variable_name(arg[i + 2]),
          normalize_variable_name(arg[i + 3]),
      };
      cached_external_field_values_.assign(3, 0.0f);
      i += 3;
      continue;
    }
    if (opt == "field9") {
      if (i + 9 >= narg) {
        error->all(FLERR,
                   "pair_style mff/torch field9 expects nine equal-style variables "
                   "(row-major: xx xy xz yx yy yz zx zy zz)");
      }
      use_external_field_ = true;
      external_tensor_rank_ = 2;
      external_field_symmetric_rank2_ = false;
      external_field_var_names_.clear();
      for (int k = 1; k <= 9; ++k) external_field_var_names_.push_back(normalize_variable_name(arg[i + k]));
      cached_external_field_values_.assign(9, 0.0f);
      i += 9;
      continue;
    }
    if (opt == "field6") {
      if (i + 6 >= narg) {
        error->all(FLERR,
                   "pair_style mff/torch field6 expects six equal-style variables "
                   "(symmetric order: xx yy zz xy xz yz)");
      }
      use_external_field_ = true;
      external_tensor_rank_ = 2;
      external_field_symmetric_rank2_ = true;
      external_field_var_names_.clear();
      for (int k = 1; k <= 6; ++k) external_field_var_names_.push_back(normalize_variable_name(arg[i + k]));
      cached_external_field_values_.assign(6, 0.0f);
      i += 6;
      continue;
    }
    error->all(FLERR, ("Unknown pair_style mff/torch option: " + opt).c_str());
  }
}

void PairMFFTorch::coeff(int narg, char **arg) {
  if (!allocated) allocate();
  if (narg < 3) error->all(FLERR, "Illegal pair_coeff command for mff/torch");

  // Expect: pair_coeff * * core.pt <elem1> <elem2> ... (ntypes entries)
  // arg[0], arg[1] are * *
  core_pt_path_ = std::string(arg[2]);

  const int ntypes = atom->ntypes;
  if (narg != 3 + ntypes) error->all(FLERR, "pair_coeff mff/torch expects one element symbol per atom type");

  type2Z_.assign(ntypes + 1, 0);
  for (int itype = 1; itype <= ntypes; itype++) {
    const std::string sym(arg[2 + itype]);
    if (sym == "NULL" || sym == "null") {
      type2Z_[itype] = 0;
      continue;
    }
    int Z = mfftorch::symbol_to_Z(sym);
    if (Z <= 0) error->all(FLERR, ("Unknown element symbol in pair_coeff mff/torch: " + sym).c_str());
    type2Z_[itype] = static_cast<int64_t>(Z);
  }

  for (int i = 1; i <= ntypes; i++) {
    for (int j = i; j <= ntypes; j++) {
      setflag[i][j] = 1;
      cutsq[i][j] = cutsq_global_;
      setflag[j][i] = 1;
      cutsq[j][i] = cutsq_global_;
    }
  }

  if (!engine_) engine_ = std::make_unique<mfftorch::MFFTorchEngine>();
  engine_loaded_ = false;  // lazy load at init_style/compute
}

void PairMFFTorch::init_style() {
  if (core_pt_path_.empty()) error->all(FLERR, "pair_coeff for mff/torch must specify core.pt path");

  // Request a full neighbor list.
  neighbor->add_request(this, NeighConst::REQ_FULL);

  try {
    if (!engine_) engine_ = std::make_unique<mfftorch::MFFTorchEngine>();
    engine_->load_core(core_pt_path_, device_str_);
    validate_external_field_configuration();
    engine_loaded_ = true;
    engine_->warmup(32, 256);
  } catch (const std::exception &e) {
    error->all(FLERR, (std::string("Failed to load TorchScript core: ") + e.what()).c_str());
  }
}

double PairMFFTorch::init_one(int i, int j) {
  return cut_global_;
}

void PairMFFTorch::validate_external_field_configuration() {
  if (!engine_) return;

  if (use_external_field_) {
    if (!engine_->accepts_external_tensor()) {
      error->all(FLERR,
                 "pair_style mff/torch field was specified, but core.pt does not accept external_tensor");
    }
    const int expected_nvars = (external_tensor_rank_ == 1) ? 3 : (external_field_symmetric_rank2_ ? 6 : 9);
    if (static_cast<int>(external_field_var_names_.size()) != expected_nvars) {
      error->all(FLERR, "mff/torch external field variable count does not match the selected field mode");
    }
    for (const auto &name : external_field_var_names_) {
      if (name.empty()) error->all(FLERR, "pair_style mff/torch external field variable name is empty");
      const int ivar = input->variable->find(name.c_str());
      if (ivar < 0) {
        error->all(FLERR, ("Unknown LAMMPS variable for mff/torch field: " + name).c_str());
      }
      if (!input->variable->equalstyle(ivar)) {
        error->all(FLERR,
                   ("mff/torch field variables must be equal-style scalars: " + name).c_str());
      }
    }
  } else if (engine_->accepts_external_tensor()) {
    error->all(FLERR,
               "core.pt requires external_tensor, but pair_style mff/torch was not given field/field6/field9");
  }
}

torch::Tensor PairMFFTorch::current_external_tensor(const torch::Device& device) {
  if (!use_external_field_) return torch::Tensor();

  std::vector<float> values(external_field_var_names_.size(), 0.0f);
  for (size_t k = 0; k < external_field_var_names_.size(); ++k) {
    const int ivar = input->variable->find(external_field_var_names_[k].c_str());
    if (ivar < 0) {
      error->all(FLERR, ("Unknown LAMMPS variable for mff/torch field: " + external_field_var_names_[k]).c_str());
    }
    values[k] = static_cast<float>(input->variable->compute_equal(ivar));
  }

  const bool cache_hit =
      external_tensor_cache_.defined() &&
      external_tensor_cache_.device() == device &&
      cached_external_field_values_ == values;
  if (cache_hit) return external_tensor_cache_;

  cached_external_field_values_ = values;
  torch::Tensor cpu;
  if (external_tensor_rank_ == 1) {
    cpu = torch::tensor(values, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  } else if (external_tensor_rank_ == 2 && external_field_symmetric_rank2_) {
    cpu = torch::tensor(
              {
                  values[0], values[3], values[4],
                  values[3], values[1], values[5],
                  values[4], values[5], values[2],
              },
              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
              .reshape({3, 3});
  } else if (external_tensor_rank_ == 2) {
    cpu = torch::tensor(values, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
              .reshape({3, 3});
  } else {
    error->all(FLERR, "Unsupported external tensor rank for mff/torch");
  }
  external_tensor_cache_ = (device.is_cpu()) ? cpu : cpu.to(device);
  return external_tensor_cache_;
}

void PairMFFTorch::reset_physical_outputs() {
  global_phys_cpu_ = torch::Tensor();
  atom_phys_cpu_ = torch::Tensor();
  global_phys_mask_cpu_ = torch::Tensor();
  atom_phys_mask_cpu_ = torch::Tensor();
  cached_phys_timestep_ = update ? static_cast<int64_t>(update->ntimestep) : -1;
}

void PairMFFTorch::cache_physical_outputs(const mfftorch::MFFOutputs& out, int nlocal) {
  cached_phys_timestep_ = update ? static_cast<int64_t>(update->ntimestep) : -1;

  if (out.global_phys.defined()) {
    global_phys_cpu_ = out.global_phys.to(torch::kCPU, torch::kFloat64).contiguous();
  } else {
    global_phys_cpu_ = torch::Tensor();
  }
  if (out.atom_phys.defined()) {
    auto atom_phys = out.atom_phys.to(torch::kCPU, torch::kFloat64).contiguous();
    if (atom_phys.dim() >= 2 && atom_phys.size(0) >= nlocal) {
      atom_phys_cpu_ = atom_phys.narrow(0, 0, nlocal).clone();
    } else {
      atom_phys_cpu_ = atom_phys.clone();
    }
  } else {
    atom_phys_cpu_ = torch::Tensor();
  }
  global_phys_mask_cpu_ = out.global_phys_mask.defined()
                              ? out.global_phys_mask.to(torch::kCPU, torch::kFloat64).contiguous()
                              : torch::Tensor();
  atom_phys_mask_cpu_ = out.atom_phys_mask.defined()
                            ? out.atom_phys_mask.to(torch::kCPU, torch::kFloat64).contiguous()
                            : torch::Tensor();
}

void PairMFFTorch::compute(int eflag, int vflag) {
  ev_init(eflag, vflag);
  reset_physical_outputs();

  if (!engine_loaded_) init_style();

  const int nlocal = atom->nlocal;
  const int nghost = atom->nghost;
  const int ntotal = nlocal + nghost;
  if (nlocal == 0) return;

  // Neighbor list
  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;

  // Build type->Z mapped A (CPU then move to engine device).
  std::vector<int64_t> A_cpu(ntotal);
  for (int i = 0; i < ntotal; i++) {
    const int itype = type[i];
    const int64_t Z = (itype >= 0 && itype < static_cast<int>(type2Z_.size())) ? type2Z_[itype] : 0;
    A_cpu[i] = Z;
  }

  // Count edges (upper bound) and build edges + rij.
  int64_t Emax = 0;
  for (int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    Emax += numneigh[i];
  }
  std::vector<int64_t> edge_src_cpu;
  std::vector<int64_t> edge_dst_cpu;
  std::vector<float> rij_cpu;
  edge_src_cpu.reserve(static_cast<size_t>(Emax));
  edge_dst_cpu.reserve(static_cast<size_t>(Emax));
  rij_cpu.reserve(static_cast<size_t>(Emax) * 3);

  for (int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    int jnum = numneigh[i];
    int *jlist = firstneigh[i];
    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj] & NEIGHMASK;

      double delx = x[j][0] - x[i][0];
      double dely = x[j][1] - x[i][1];
      double delz = x[j][2] - x[i][2];
      domain->minimum_image(FLERR, delx, dely, delz);

      double rsq = delx * delx + dely * dely + delz * delz;
      if (rsq > cutsq_global_) continue;

      edge_src_cpu.push_back(static_cast<int64_t>(i));
      edge_dst_cpu.push_back(static_cast<int64_t>(j));
      rij_cpu.push_back(static_cast<float>(delx));
      rij_cpu.push_back(static_cast<float>(dely));
      rij_cpu.push_back(static_cast<float>(delz));
    }
  }

  const int64_t E = static_cast<int64_t>(edge_src_cpu.size());
  if (E <= 1) return;

  // Torch tensors (CPU -> device copy).
  auto A_t = torch::from_blob(A_cpu.data(), {ntotal}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU)).clone();
  auto edge_src_t = torch::from_blob(edge_src_cpu.data(), {E}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU)).clone();
  auto edge_dst_t = torch::from_blob(edge_dst_cpu.data(), {E}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU)).clone();
  auto rij_t = torch::from_blob(rij_cpu.data(), {E, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone();
  auto external_tensor_t = current_external_tensor(torch::kCPU);

  const bool want_atom_virial = static_cast<bool>(vflag_atom);
  mfftorch::MFFOutputs out;
  try {
    out = engine_->compute(nlocal, ntotal, A_t, edge_src_t, edge_dst_t, rij_t, external_tensor_t,
                           static_cast<bool>(eflag), want_atom_virial);
  } catch (const std::exception &e) {
    error->all(FLERR, (std::string("mff/torch engine compute failed: ") + e.what()).c_str());
  }
  cache_physical_outputs(out, nlocal);

  if (eflag) eng_vdwl += out.energy;

  // When virial is needed, ghost forces must be in f[] for virial_fdotr_compute()
  // to produce correct results (it sums over nall = nlocal + nghost).
  const int nwrite = (force->newton_pair || vflag_fdotr) ? ntotal : nlocal;
  auto forces_cpu = out.forces.to(torch::kCPU, torch::kFloat64).contiguous();
  const double *fp = forces_cpu.data_ptr<double>();
  for (int i = 0; i < nwrite; i++) {
    f[i][0] += fp[i * 3 + 0];
    f[i][1] += fp[i * 3 + 1];
    f[i][2] += fp[i * 3 + 2];
  }

  if (eflag_atom && eatom && out.atom_energy.defined()) {
    auto ae_cpu = out.atom_energy.to(torch::kCPU, torch::kFloat64).contiguous().view({ntotal});
    const double *ep = ae_cpu.data_ptr<double>();
    for (int i = 0; i < nlocal; i++) eatom[i] += ep[i];
  }

  if (vflag_atom && vatom && out.atom_virial.defined()) {
    auto vir_cpu = out.atom_virial.to(torch::kCPU, torch::kFloat64).contiguous();
    const double *vp = vir_cpu.data_ptr<double>();
    const int nvir = force->newton_pair ? ntotal : nlocal;
    for (int i = 0; i < nvir; i++) {
      vatom[i][0] += vp[i * 6 + 0];
      vatom[i][1] += vp[i * 6 + 1];
      vatom[i][2] += vp[i * 6 + 2];
      vatom[i][3] += vp[i * 6 + 3];
      vatom[i][4] += vp[i * 6 + 4];
      vatom[i][5] += vp[i * 6 + 5];
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}
