#pragma once

#include <mpi.h>

#include "krapivin_a_ring/common/include/common.hpp"
#include "task/include/task.hpp"

namespace krapivin_a_ring {

class KrapivinARingMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KrapivinARingMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  // Добавляем приватные методы
  void HandleSource(MPI_Comm ring_comm, int ring_rank, int next_rank, int target, int data);
  void HandleParticipant(MPI_Comm ring_comm, int prev_rank, int next_rank, int ring_rank, int target);

  // Статические вспомогательные функции
  static void AddDelay();
  static bool ComputeIsParticipant(int ring_rank, int source, int target);
};

}  // namespace krapivin_a_ring
