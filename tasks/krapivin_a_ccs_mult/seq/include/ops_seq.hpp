#pragma once

#include "krapivin_a_ccs_mult/common/include/common.hpp"
#include "task/include/task.hpp"

namespace krapivin_a_ccs_mult {

class KrapivinACcsMultSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit KrapivinACcsMultSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace krapivin_a_ccs_mult
