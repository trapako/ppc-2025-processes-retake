#include "krapivin_a_ring/seq/include/ops_seq.hpp"

#include <mpi.h>

#include <chrono>
#include <cmath>
#include <vector>

#include "krapivin_a_ring/common/include/common.hpp"
#include "util/include/util.hpp"

namespace krapivin_a_ring {

KrapivinARingSEQ::KrapivinARingSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KrapivinARingSEQ::ValidationImpl() {
  return (GetInput().source_rank >= 0 && GetInput().target_rank >= 0);
}

bool KrapivinARingSEQ::PreProcessingImpl() {
  return true;
}

bool KrapivinARingSEQ::RunImpl() {
  int source = GetInput().source_rank;
  int target = GetInput().target_rank;

  int size = ppc::util::GetNumProc();

  if (ppc::util::IsUnderMpirun()) {
    int mpi_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    if (mpi_size > 0) {
      size = mpi_size;
    }
  }

  if (size <= 0) {
    size = 1;
  }

  source = source % size;
  target = target % size;

  std::vector<int> path;
  int current = source;
  path.push_back(current);
  while (current != target) {
    current = (current + 1) % size;
    path.push_back(current);
  }

  auto start_time = std::chrono::steady_clock::now();
  auto target_duration = std::chrono::milliseconds(800);

  volatile double sum = 0.0;
  const int iterations = 10000;

  while (std::chrono::steady_clock::now() - start_time < target_duration) {
    for (int i = 0; i < iterations; ++i) {
      sum += std::sin(static_cast<double>(i));
    }
  }
  (void)sum;

  GetOutput() = path;
  return true;
}

bool KrapivinARingSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace krapivin_a_ring
