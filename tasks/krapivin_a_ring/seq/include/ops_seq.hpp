#pragma once

#include "krapivin_a_ring/common/include/common.hpp"
#include "task/include/task.hpp"

namespace krapivin_a_ring {

class KrapivinARingSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit KrapivinARingSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace krapivin_a_ring
