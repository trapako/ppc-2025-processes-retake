#include <gtest/gtest.h>

#include <limits>
#include <random>

#include "krapivin_a_ccs_mult/common/include/common.hpp"
#include "krapivin_a_ccs_mult/mpi/include/ops_mpi.hpp"
#include "krapivin_a_ccs_mult/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace krapivin_a_ccs_mult {

class KrapivinACcsMultPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int k_count_ = 100000;
  int correct_data_ = 0;
  InType input_data_;

  void SetUp() override {
    
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::get<2>(output_data).empty();
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(KrapivinACcsMultPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, KrapivinACcsMultMPI, KrapivinACcsMultSEQ>(
    PPC_SETTINGS_krapivin_a_ccs_mult);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KrapivinACcsMultPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KrapivinACcsMultPerfTest, kGtestValues, kPerfTestName);

}  // namespace krapivin_a_ccs_mult
