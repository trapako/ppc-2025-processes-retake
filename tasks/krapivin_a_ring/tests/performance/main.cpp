#include <gtest/gtest.h>
#include <mpi.h>

#include <chrono>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "krapivin_a_ring/common/include/common.hpp"
#include "krapivin_a_ring/mpi/include/ops_mpi.hpp"
#include "krapivin_a_ring/seq/include/ops_seq.hpp"
#include "performance/include/performance.hpp"
#include "task/include/task.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace krapivin_a_ring {

namespace {
OutType CalculateExpectedPath(const InType &input, int size) {
  std::vector<int> path_history;
  int current_rank = input.source_rank;
  int target_rank = input.target_rank;

  if (size <= 0) {
    return path_history;
  }

  int max_steps = size;
  int steps = 0;

  while (current_rank != target_rank && steps < max_steps) {
    path_history.push_back(current_rank);
    current_rank = (current_rank + 1) % size;
    steps++;
  }

  if (steps < max_steps || current_rank == target_rank) {
    path_history.push_back(current_rank);
  }

  return path_history;
}
}  // namespace

class KrapivinARingPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_{};
  bool is_seq_test_ = false;

  void SetUp() override {  // rerun ci comment
    int size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int source_rank = 0;
    int target_rank = (size > 0) ? (size - 1) : 0;
    input_data_ = RingTaskData{100, source_rank, target_rank};

    const auto &param = GetParam();
    const auto &name = std::get<1>(param);
    is_seq_test_ = (name.find("_seq_") != std::string::npos);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int normalized_target = input_data_.target_rank;
    if (size > 0) {
      normalized_target = normalized_target % size;
    }

    if (ppc::util::IsUnderMpirun() && is_seq_test_) {
      if (rank != normalized_target) {
        return true;
      }
    }

    if (output_data.empty()) {
      return rank != normalized_target;
    }

    int normalized_source = input_data_.source_rank;
    if (size > 0) {
      normalized_source = normalized_source % size;
    }

    RingTaskData normalized_input = input_data_;
    normalized_input.source_rank = normalized_source;
    normalized_input.target_rank = normalized_target;

    OutType expected_path = CalculateExpectedPath(normalized_input, size);

    if (output_data.size() != expected_path.size()) {
      return false;
    }

    for (size_t i = 0; i < output_data.size(); ++i) {
      if (output_data[i] != expected_path[i]) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  void RunPerfAndReport(const std::function<ppc::task::TaskPtr<InType, OutType>(InType)> &task_getter,
                        const std::string &test_name, ppc::performance::PerfResults::TypeOfRunning mode) {
    auto task = task_getter(GetTestInputData());
    ppc::performance::Perf<InType, OutType> perf(task);
    ppc::performance::PerfAttr perf_attr;

    const auto t0 = std::chrono::high_resolution_clock::now();
    perf_attr.current_timer = [t0] {
      auto now = std::chrono::high_resolution_clock::now();
      auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now - t0).count();
      return static_cast<double>(ns) * 1e-9;
    };

    if (mode == ppc::performance::PerfResults::TypeOfRunning::kPipeline) {
      perf.PipelineRun(perf_attr);
    } else {
      perf.TaskRun(perf_attr);
    }

    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    double time_to_report = perf.GetPerfResults().time_sec;
    if (mpi_rank == 0) {
      perf.PrintPerfStatistic(test_name);
    } else {
      MPI_Send(&time_to_report, 1, MPI_DOUBLE, 0, 12345, MPI_COMM_WORLD);
    }

    OutType output_data = task->GetOutput();
    ASSERT_TRUE(CheckTestOutputData(output_data));
  }

  static void ReceiveAndPrintPerfResult(const std::string &test_name, ppc::performance::PerfResults::TypeOfRunning mode,
                                        int source_rank) {
    double recv_time = 0.0;
    MPI_Recv(&recv_time, 1, MPI_DOUBLE, source_rank, 12345, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    std::stringstream perf_res_str;
    perf_res_str << std::fixed << std::setprecision(10) << recv_time;
    std::string type_test_name =
        (mode == ppc::performance::PerfResults::TypeOfRunning::kTaskRun) ? "task_run" : "pipeline";
    std::cout << test_name << ":" << type_test_name << ":" << perf_res_str.str() << '\n';
  }

 protected:
  void ExecuteTest(const ppc::util::PerfTestParam<InType, OutType> &perf_test_param) {
    auto task_getter = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTaskGetter)>(perf_test_param);
    auto test_name = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kNameTest)>(perf_test_param);
    auto mode = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(perf_test_param);

    if (!ppc::util::IsUnderMpirun() || !is_seq_test_) {
      BaseRunPerfTests<InType, OutType>::ExecuteTest(perf_test_param);
      return;
    }

    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    int normalized_target = (mpi_size > 0) ? (input_data_.target_rank % mpi_size) : input_data_.target_rank;

    if (mpi_rank == normalized_target) {
      RunPerfAndReport(task_getter, test_name, mode);
    } else if (mpi_rank == 0) {
      ReceiveAndPrintPerfResult(test_name, mode, normalized_target);
    }
  }
};

namespace {
TEST_P(KrapivinARingPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, KrapivinARingMPI, KrapivinARingSEQ>(PPC_SETTINGS_krapivin_a_ring);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KrapivinARingPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KrapivinARingPerfTest, kGtestValues, kPerfTestName);
}  // namespace
}  // namespace krapivin_a_ring
