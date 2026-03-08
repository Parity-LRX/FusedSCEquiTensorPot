#include "pair_mff_torch_kokkos.h"

#ifdef LMP_KOKKOS

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
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
  if (core_pt_path_.empty()) error->all(FLERR, "pair_coeff for mff/torch must specify core.pt path");

  // Request a full neighbor list (same as base class).
  neighbor->add_request(this, NeighConst::REQ_FULL);

  neighflag = lmp->kokkos->neighflag;
  auto request = neighbor->find_request(this);
  request->set_kokkos_host(std::is_same_v<DeviceType, LMPHostType> && !std::is_same_v<DeviceType, LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType, LMPDeviceType>);
  if (neighflag == FULL) request->enable_full();

  if (!engine_) engine_ = std::make_unique<mfftorch::MFFTorchEngine>();
  if (!engine_loaded_) {
    // Each MPI rank uses its own GPU: gpu_id = local_rank % num_gpus.
    std::string dev = device_str_;
    if (dev == "cuda") {
      int ngpus = lmp->kokkos->ngpus;
      int gpu_id = 0;
      if (ngpus > 0) {
        gpu_id = comm->me % ngpus;
      }
      dev = "cuda:" + std::to_string(gpu_id);
    }
    try {
      engine_->load_core(core_pt_path_, dev);
      validate_external_field_configuration();
      engine_loaded_ = true;
    } catch (const std::exception &e) {
      error->all(FLERR, (std::string("Failed to load TorchScript core: ") + e.what()).c_str());
    }
  }

  if (engine_->is_cuda()) {
    auto t = torch::from_blob(type2Z_.data(), {(int64_t)type2Z_.size()},
                              torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU))
                 .clone();
    type2Z_cuda_ = t.to(engine_->device());

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
  // Use default allocation behavior so eatom/vatom are available when
  // computes like pe/atom or stress/atom request per-atom quantities.
  ev_init(eflag, vflag);
  reset_physical_outputs();

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

  auto dev = engine_->device();
  if (cached_Etotal_ != Etotal) {
    buf_edge_src_ = torch::empty({Etotal}, torch::TensorOptions().dtype(torch::kInt64).device(dev));
    buf_edge_dst_ = torch::empty({Etotal}, torch::TensorOptions().dtype(torch::kInt64).device(dev));
    buf_rij_ = torch::empty({Etotal, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    cached_Etotal_ = Etotal;
  }
  if (cached_ntotal_ != static_cast<int64_t>(ntotal)) {
    buf_type_idx_ = torch::empty({ntotal}, torch::TensorOptions().dtype(torch::kInt64).device(dev));
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
  auto external_tensor = current_external_tensor(dev);

  const bool need_energy = static_cast<bool>(eflag_global || eflag_atom);
  const bool need_atom_virial = static_cast<bool>(vflag_atom);
  mfftorch::MFFOutputs out;
  try {
    out = engine_->compute(nlocal, ntotal, A, buf_edge_src_, buf_edge_dst_, buf_rij_, external_tensor,
                           need_energy, need_atom_virial);
  } catch (const std::exception &e) {
    error->all(FLERR, (std::string("mff/torch/kk engine compute failed: ") + e.what()).c_str());
  }
  cache_physical_outputs(out, nlocal);

  if (eflag_global) eng_vdwl += out.energy;

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

  // Per-atom energy: copy NN atom_energy to LAMMPS eatom (local atoms only).
  if (eflag_atom && eatom && out.atom_energy.defined()) {
    auto ae = out.atom_energy.to(torch::kCPU, torch::kFloat64).contiguous().view({ntotal});
    const double *ep = ae.data_ptr<double>();
    for (int i = 0; i < nlocal; i++) eatom[i] += ep[i];
  }

  // Per-atom virial: engine computed atom_virial [ntotal, 6] on GPU via edge-force
  // outer products (rij ⊗ edge_forces), scatter-added 50/50 to src and dst.
  // With newton OFF + FULL list, only write LOCAL atoms (consistent with ev_tally
  // convention — each pair is visited twice, so each local atom gets its full share).
  if (vflag_atom && vatom && out.atom_virial.defined()) {
    auto avir = out.atom_virial.to(torch::kCPU, torch::kFloat64).contiguous();
    const double *vp = avir.data_ptr<double>();
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

#ifdef MFF_ENABLE_VIRIAL
  if (vflag_global) {
    // f·r virial must sum over ALL atoms (local + ghost).
    // Ghost atoms carry the "other half" of cross-boundary PBC interactions.
    // Using forces_v (NN output) directly guarantees ghost forces are included.
    Kokkos::View<double[6], DeviceType> d_virial("mfftorch::d_virial");
    Kokkos::deep_copy(d_virial, 0.0);

    Kokkos::parallel_for(
        "mfftorch::virial_fdotr", ntotal, KOKKOS_LAMBDA(const int i) {
          const double xi = x(i, 0), yi = x(i, 1), zi = x(i, 2);
          const double fx = static_cast<double>(forces_v(i, 0));
          const double fy = static_cast<double>(forces_v(i, 1));
          const double fz = static_cast<double>(forces_v(i, 2));
          Kokkos::atomic_add(&d_virial(0), fx * xi);  // xx
          Kokkos::atomic_add(&d_virial(1), fy * yi);  // yy
          Kokkos::atomic_add(&d_virial(2), fz * zi);  // zz
          Kokkos::atomic_add(&d_virial(3), fy * xi);  // xy
          Kokkos::atomic_add(&d_virial(4), fz * xi);  // xz
          Kokkos::atomic_add(&d_virial(5), fz * yi);  // yz
        });
    Kokkos::fence();

    auto h_virial = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_virial);
    for (int n = 0; n < 6; n++) virial[n] += h_virial(n);
  }
#endif
}

namespace LAMMPS_NS {
template class PairMFFTorchKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairMFFTorchKokkos<LMPHostType>;
#endif
}  // namespace LAMMPS_NS

#endif  // LMP_KOKKOS
