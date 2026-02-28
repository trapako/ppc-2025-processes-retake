#include "krapivin_a_ring/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "krapivin_a_ring/common/include/common.hpp"

namespace krapivin_a_ring {

void KrapivinARingMPI::AddDelay() {
  auto start_time = std::chrono::steady_clock::now();
  auto target_duration = std::chrono::milliseconds(200);

  volatile double sum = 0.0;
  const int iterations = 10000;

  while (std::chrono::steady_clock::now() - start_time < target_duration) {
    for (int i = 0; i < iterations; ++i) {
      sum += std::sin(static_cast<double>(i));
    }
  }
  (void)sum;
}

bool KrapivinARingMPI::ComputeIsParticipant(int ring_rank, int source, int target) {
  if (source == target) {
    return ring_rank == source;
  }
  if (source < target) {
    return (ring_rank >= source && ring_rank <= target);
  }
  return (ring_rank >= source || ring_rank <= target);
}

void KrapivinARingMPI::HandleSource(MPI_Comm ring_comm, int ring_rank, int next_rank, int target, int data) {
  std::vector<int> path_history;
  path_history.push_back(ring_rank);
  AddDelay();
  if (ring_rank == target) {
    GetOutput() = std::move(path_history);
    return;
  }
  int path_size = static_cast<int>(path_history.size());
  MPI_Send(&path_size, 1, MPI_INT, next_rank, 0, ring_comm);
  if (path_size > 0) {
    MPI_Send(path_history.data(), path_size, MPI_INT, next_rank, 1, ring_comm);
  }
  MPI_Send(&data, 1, MPI_INT, next_rank, 2, ring_comm);
}

void KrapivinARingMPI::HandleParticipant(MPI_Comm ring_comm, int prev_rank, int next_rank, int ring_rank, int target) {
  int path_size = 0;
  MPI_Status status;
  MPI_Recv(&path_size, 1, MPI_INT, prev_rank, 0, ring_comm, &status);
  int comm_size = 0;
  MPI_Comm_size(ring_comm, &comm_size);
  if (path_size < 0) {
    path_size = 0;
  } else if (path_size > comm_size) {
    path_size = comm_size;
  }
  std::vector<int> path_history(static_cast<std::size_t>(path_size));
  if (path_size > 0) {
    MPI_Recv(path_history.data(), path_size, MPI_INT, prev_rank, 1, ring_comm, &status);
  }
  int received_data = 0;
  MPI_Recv(&received_data, 1, MPI_INT, prev_rank, 2, ring_comm, &status);
  AddDelay();
  path_history.push_back(ring_rank);
  if (ring_rank == target) {
    GetOutput() = path_history;
    return;
  }
  path_size = static_cast<int>(path_history.size());
  MPI_Send(&path_size, 1, MPI_INT, next_rank, 0, ring_comm);
  MPI_Send(path_history.data(), path_size, MPI_INT, next_rank, 1, ring_comm);
  MPI_Send(&received_data, 1, MPI_INT, next_rank, 2, ring_comm);
}

KrapivinARingMPI::KrapivinARingMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KrapivinARingMPI::ValidationImpl() {
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized == 0) {
    return (GetInput().source_rank >= 0 && GetInput().target_rank >= 0);
  }
  return (GetInput().source_rank >= 0 && GetInput().target_rank >= 0);
}

bool KrapivinARingMPI::PreProcessingImpl() {
  return true;
}

bool KrapivinARingMPI::RunImpl() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const auto &input = GetInput();
  int source = input.source_rank;
  int target = input.target_rank;

  if (world_size > 0) {
    source = source % world_size;
    target = target % world_size;
  }

  MPI_Group world_group = MPI_GROUP_NULL;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);
  MPI_Comm ring_comm = MPI_COMM_WORLD;
  MPI_Comm_dup(MPI_COMM_WORLD, &ring_comm);

  int ring_rank = 0;
  int ring_size = 0;
  MPI_Comm_rank(ring_comm, &ring_rank);
  MPI_Comm_size(ring_comm, &ring_size);

  int next_rank = (ring_rank + 1) % ring_size;
  int prev_rank = (ring_rank - 1 + ring_size) % ring_size;

  MPI_Group_free(&world_group);

  bool is_participant = ComputeIsParticipant(ring_rank, source, target);

  if (ring_rank == source) {
    // ИСПРАВЛЕНО: убрали this из вызова
    HandleSource(ring_comm, ring_rank, next_rank, target, input.data);
  } else if (is_participant) {
    // ИСПРАВЛЕНО: убрали this из вызова
    HandleParticipant(ring_comm, prev_rank, next_rank, ring_rank, target);
  } else {
    // Non-participant does nothing
  }

  MPI_Barrier(ring_comm);
  MPI_Comm_free(&ring_comm);
  return true;
}

bool KrapivinARingMPI::PostProcessingImpl() {
  return true;
}

}  // namespace krapivin_a_ring
