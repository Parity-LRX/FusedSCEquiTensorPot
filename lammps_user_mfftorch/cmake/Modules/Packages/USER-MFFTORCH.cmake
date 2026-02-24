# USER-MFFTORCH package: LibTorch-powered pair style(s)

find_package(Torch REQUIRED)

# Core (always)
target_sources(lammps PRIVATE
  ${LAMMPS_SOURCE_DIR}/USER-MFFTORCH/mff_torch_engine.cpp
  ${LAMMPS_SOURCE_DIR}/USER-MFFTORCH/pair_mff_torch.cpp
)

# Kokkos variant (only when KOKKOS package is enabled)
if(PKG_KOKKOS)
  target_sources(lammps PRIVATE
    ${LAMMPS_SOURCE_DIR}/USER-MFFTORCH/pair_mff_torch_kokkos.cpp
  )
endif()

target_include_directories(lammps PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(lammps PRIVATE "${TORCH_LIBRARIES}")

if(TORCH_CXX_FLAGS)
  separate_arguments(_TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
  target_compile_options(lammps PRIVATE ${_TORCH_CXX_FLAGS_LIST})
endif()

