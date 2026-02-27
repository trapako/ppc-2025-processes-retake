// ops_mpi.hpp
#pragma once

#include <vector>

#include "krapivin_a_ccs_mult/common/include/common.hpp"
#include "task/include/task.hpp"

namespace krapivin_a_ccs_mult {

class KrapivinACcsMultMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KrapivinACcsMultMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void PrintCCS(const ccs& m);
  // Входные матрицы в формате CCS
  ccs A_;
  ccs B_;

  // Локальные данные для параллельного умножения
  std::vector<double> local_result_;    // локальная плотная матрица (A.rows x local_cols_count_)
};

}  // namespace krapivin_a_ccs_mult