#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <string>
#include <tuple>

#include "krapivin_a_ring/common/include/common.hpp"
#include "krapivin_a_ring/mpi/include/ops_mpi.hpp"
#include "krapivin_a_ring/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace krapivin_a_ring {

using FuncTestType = std::tuple<InType, std::string>;

namespace {

OutType CalculateExpectedRingPath(const InType &input, int size) {
  OutType expected_path;
  if (size <= 0) {
    return expected_path;
  }

  int current = input.source_rank % size;
  int target = input.target_rank % size;

  expected_path.push_back(current);
  while (current != target) {
    current = (current + 1) % size;
    expected_path.push_back(current);
  }
  return expected_path;
}
}  // namespace

class KrapivinARingFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, FuncTestType> {
 public:
  static std::string PrintTestParam(const FuncTestType &test_param) {
    const auto &input = std::get<0>(test_param);
    const auto &desc = std::get<1>(test_param);
    return "From_" + std::to_string(input.source_rank) + "_To_" + std::to_string(input.target_rank) + "_" + desc;
  }

  bool CheckTestOutputData(OutType &output_data) override {
    int rank = 0;
    int size = 1;

    if (ppc::util::IsUnderMpirun()) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &size);
    } else {
      size = ppc::util::GetNumProc();
    }

    int target = input_data_.target_rank % size;

    if (rank != target) {
      return true;
    }
    if (output_data.empty()) {
      return false;
    }

    OutType expected = CalculateExpectedRingPath(input_data_, size);
    return (output_data == expected);
  }

  InType GetTestInputData() override {
    return input_data_;
  }

 protected:
  void SetUp() override {
    const auto &full_params = GetParam();
    const auto &user_test_data = std::get<2>(full_params);
    input_data_ = std::get<0>(user_test_data);
  }

 private:
  InType input_data_{};
};

namespace {
TEST_P(KrapivinARingFuncTests, RingPathVerification) {
  ExecuteTest(GetParam());
}

const std::array<FuncTestType, 5> kRingTestParams = {
    FuncTestType({100, 0, 2}, "ForwardShort"), FuncTestType({200, 2, 0}, "WrapAround"),
    FuncTestType({300, 1, 1}, "SelfDelivery"), FuncTestType({400, 0, 3}, "LongPath"),
    FuncTestType({500, 3, 2}, "AlmostFullCircle")};

const auto kFuncTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<KrapivinARingMPI, InType>(kRingTestParams, PPC_SETTINGS_krapivin_a_ring),
                   ppc::util::AddFuncTask<KrapivinARingSEQ, InType>(kRingTestParams, PPC_SETTINGS_krapivin_a_ring));

INSTANTIATE_TEST_SUITE_P(KrapivinARingTests, KrapivinARingFuncTests, ppc::util::TupleToGTestValues(kFuncTasksList),
                         KrapivinARingFuncTests::PrintFuncTestName<KrapivinARingFuncTests>);
}  // namespace
}  // namespace krapivin_a_ring
