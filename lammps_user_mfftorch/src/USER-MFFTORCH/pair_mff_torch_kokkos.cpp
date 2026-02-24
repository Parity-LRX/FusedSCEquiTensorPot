#include "pair_mff_torch_kokkos.h"

#ifdef LMP_KOKKOS

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "neigh_list.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "neighbor_kokkos.h"

#include "mff_torch_engine.h"

#include <Kokkos_Core.hpp>
#include <type_traits>

using namespace LAMMPS_NS;

template <class DeviceType>
PairMFFTorchKokkos<DeviceType>::PairMFFTorchKokkos(LAMMPS *lmp) : PairMFFTorch(lmp) {
  kokkosable = 1;
  atomKK = (AtomKokkos *)atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
}

template <class DeviceType>
void PairMFFTorchKokkos<DeviceType>::init_style() {
  PairMFFTorch::init_style();

  neighflag = lmp->kokkos->neighflag;
  auto request = neighbor->find_request(this);
  request->set_kokkos_host(std::is_same_v<DeviceType, LMPHostType> && !std::is_same_v<DeviceType, LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType, LMPDeviceType>);
  if (neighflag == FULL) request->enable_full();

  if (!engine_) engine_ = std::make_unique<mfftorch::MFFTorchEngine>();
  if (!engine_loaded_) {
    try {
      engine_->load_core(core_pt_path_, device_str_);
      engine_loaded_ = true;
    } catch (const std::exception &e) {
      error->all(FLERR, (std::string("Failed to load TorchScript core: ") + e.what()).c_str());
    }
  }

  if (engine_->is_cuda()) {
    auto t = torch::from_blob(type2Z_.data(), {(int64_t)type2Z_.size()},
                              torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU))
                 .clone();
    type2Z_cuda_ = t.to(torch::kCUDA);

    engine_->warmup(32, 256);
  }
}

template <class DeviceType>
void PairMFFTorchKokkos<DeviceType>::compute(int eflag_in, int vflag_in) {
  if (!engine_loaded_) init_style();
  if (!engine_ || !engine_->is_cuda()) {
    PairMFFTorch::compute(eflag_in, vflag_in);
    return;
  }

  int eflag = eflag_in;
  int vflag = vflag_in;

  if (neighflag == FULL) no_virial_fdotr_compute = 1;
  ev_init(eflag, vflag, 0);

  atomKK->sync(execution_space, X_MASK | F_MASK | TYPE_MASK);
  atomKK->modified(execution_space, F_MASK);

  auto x = atomKK->k_x.template view<DeviceType>();
  auto f = atomKK->k_f.template view<DeviceType>();
  auto type = atomKK->k_type.template view<DeviceType>();

  nlocal = atom->nlocal;
  nall = atom->nlocal + atom->nghost;
  const int ntotal = nall;

  NeighListKokkos<DeviceType> *k_list = static_cast<NeighListKokkos<DeviceType> *>(list);
  auto d_numneigh = k_list->d_numneigh;
  auto d_neighbors = k_list->d_neighbors;
  auto d_ilist = k_list->d_ilist;
  const int inum = list->inum;

  if (nlocal == 0 || inum == 0) return;

  // Count total edges via device reduce.
  int64_t Etotal = 0;
  Kokkos::parallel_reduce(
      "mfftorch::count_edges", inum,
      KOKKOS_LAMBDA(const int ii, int64_t &acc) {
        acc += static_cast<int64_t>(d_numneigh[d_ilist[ii]]);
      },
      Etotal);
  Kokkos::fence();
  if (Etotal <= 1) return;

  // Exclusive-scan offsets on device using parallel_scan.
  Kokkos::View<int64_t *, DeviceType> d_offsets("mfftorch::offsets", inum);
  Kokkos::parallel_scan(
      "mfftorch::scan_offsets", inum,
      KOKKOS_LAMBDA(const int ii, int64_t &update, const bool is_final) {
        if (is_final) d_offsets(ii) = update;
        update += static_cast<int64_t>(d_numneigh[d_ilist[ii]]);
      });
  Kokkos::fence();

  // Reuse CUDA tensor buffers when edge count / atom count unchanged.
  using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;

  if (cached_Etotal_ != Etotal) {
    buf_edge_src_ = torch::empty({Etotal}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    buf_edge_dst_ = torch::empty({Etotal}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    buf_rij_ = torch::empty({Etotal, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    cached_Etotal_ = Etotal;
  }
  if (cached_ntotal_ != static_cast<int64_t>(ntotal)) {
    buf_type_idx_ = torch::empty({ntotal}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    cached_ntotal_ = ntotal;
  }

  Kokkos::View<int64_t *, DeviceType, Unmanaged> edge_src_v(buf_edge_src_.data_ptr<int64_t>(), Etotal);
  Kokkos::View<int64_t *, DeviceType, Unmanaged> edge_dst_v(buf_edge_dst_.data_ptr<int64_t>(), Etotal);
  Kokkos::View<float **, Kokkos::LayoutRight, DeviceType, Unmanaged> rij_v(buf_rij_.data_ptr<float>(), Etotal, 3);

  Kokkos::parallel_for(
      "mfftorch::fill_edges", inum, KOKKOS_LAMBDA(const int ii) {
        const int i = d_ilist[ii];
        const int jnum = d_numneigh[i];
        const int64_t base = d_offsets(ii);
        for (int jj = 0; jj < jnum; jj++) {
          int j = d_neighbors(i, jj) & NEIGHMASK;
          const int64_t idx = base + jj;
          edge_src_v(idx) = static_cast<int64_t>(i);
          edge_dst_v(idx) = static_cast<int64_t>(j);
          rij_v(idx, 0) = static_cast<float>(x(j, 0) - x(i, 0));
          rij_v(idx, 1) = static_cast<float>(x(j, 1) - x(i, 1));
          rij_v(idx, 2) = static_cast<float>(x(j, 2) - x(i, 2));
        }
      });
  Kokkos::fence();

  // Type -> Z mapping on device.
  Kokkos::View<int64_t *, DeviceType, Unmanaged> type_idx_v(buf_type_idx_.data_ptr<int64_t>(), ntotal);
  Kokkos::parallel_for(
      "mfftorch::fill_type_idx", ntotal,
      KOKKOS_LAMBDA(const int i) { type_idx_v(i) = static_cast<int64_t>(type(i)); });
  Kokkos::fence();

  auto A = type2Z_cuda_.index_select(0, buf_type_idx_);

  const bool need_energy = static_cast<bool>(eflag_global);
  mfftorch::MFFOutputs out;
  try {
    out = engine_->compute(nlocal, ntotal, A, buf_edge_src_, buf_edge_dst_, buf_rij_, need_energy);
  } catch (const std::exception &e) {
    error->all(FLERR, (std::string("mff/torch/kk engine compute failed: ") + e.what()).c_str());
  }

  if (need_energy) eng_vdwl += out.energy;

  // Write forces on device (no host transfer).
  auto forces = out.forces.contiguous();
  Kokkos::View<float **, Kokkos::LayoutRight, DeviceType, Unmanaged> forces_v(forces.data_ptr<float>(), ntotal, 3);

  const int nwrite = force->newton_pair ? ntotal : nlocal;
  Kokkos::parallel_for(
      "mfftorch::add_forces", nwrite, KOKKOS_LAMBDA(const int i) {
        f(i, 0) += forces_v(i, 0);
        f(i, 1) += forces_v(i, 1);
        f(i, 2) += forces_v(i, 2);
      });
  Kokkos::fence();

  if (vflag_fdotr) {
    atomKK->modified(execution_space, F_MASK);
    atomKK->sync(Host, X_MASK | F_MASK);
    virial_fdotr_compute();
  }
}

namespace LAMMPS_NS {
template class PairMFFTorchKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairMFFTorchKokkos<LMPHostType>;
#endif
}  // namespace LAMMPS_NS

#endif  // LMP_KOKKOS
