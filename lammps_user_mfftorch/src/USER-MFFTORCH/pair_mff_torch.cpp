#include "pair_mff_torch.h"

#include "atom.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "utils.h"

#include "mff_periodic_table.h"
#include "mff_torch_engine.h"

#include <cmath>
#include <stdexcept>

using namespace LAMMPS_NS;

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
  if (narg < 1 || narg > 2) error->all(FLERR, "Illegal pair_style mff/torch command");
  cut_global_ = utils::numeric(FLERR, arg[0], false, lmp);
  if (cut_global_ <= 0.0) error->all(FLERR, "pair_style mff/torch cutoff must be > 0");
  cutsq_global_ = cut_global_ * cut_global_;

  if (narg == 2) {
    device_str_ = std::string(arg[1]);
    if (!(device_str_ == "cpu" || device_str_ == "cuda")) {
      error->all(FLERR, "pair_style mff/torch device must be 'cpu' or 'cuda'");
    }
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
    engine_loaded_ = true;
    engine_->warmup(32, 256);
  } catch (const std::exception &e) {
    error->all(FLERR, (std::string("Failed to load TorchScript core: ") + e.what()).c_str());
  }
}

double PairMFFTorch::init_one(int i, int j) {
  return cut_global_;
}

void PairMFFTorch::compute(int eflag, int vflag) {
  ev_init(eflag, vflag);

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

  mfftorch::MFFOutputs out;
  try {
    out = engine_->compute(nlocal, ntotal, A_t, edge_src_t, edge_dst_t, rij_t, static_cast<bool>(eflag));
  } catch (const std::exception &e) {
    error->all(FLERR, (std::string("mff/torch engine compute failed: ") + e.what()).c_str());
  }

  if (eflag) eng_vdwl += out.energy;

  // Write forces back. (StepA: device -> CPU copy)
  const int nwrite = force->newton_pair ? ntotal : nlocal;
  auto forces_cpu = out.forces.to(torch::kCPU, torch::kFloat64).contiguous();
  const double *fp = forces_cpu.data_ptr<double>();
  for (int i = 0; i < nwrite; i++) {
    f[i][0] += fp[i * 3 + 0];
    f[i][1] += fp[i * 3 + 1];
    f[i][2] += fp[i * 3 + 2];
  }

  if (eflag_atom) {
    // Add per-atom energies for owned atoms (LAMMPS expects eatom sized nlocal).
    auto ae_cpu = out.atom_energy.to(torch::kCPU, torch::kFloat64).contiguous().view({ntotal});
    const double *ep = ae_cpu.data_ptr<double>();
    for (int i = 0; i < nlocal; i++) eatom[i] += ep[i];
  }

  if (vflag_fdotr) virial_fdotr_compute();
}
