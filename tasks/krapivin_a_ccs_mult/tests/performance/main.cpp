#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "krapivin_a_ccs_mult/common/include/common.hpp"
#include "krapivin_a_ccs_mult/mpi/include/ops_mpi.hpp"
#include "krapivin_a_ccs_mult/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace krapivin_a_ccs_mult {

class KrapivinACcsMultPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const size_t k_size_ = 2000;
  const double percent_first_ = 0.01;
  const double percent_second_ = 0.01;
  std::vector<double> correct_data_;
  InType input_data_;

  void SetUp() override {
    std::vector<double> first = GenerateDenseMatrix(k_size_, k_size_, percent_first_, 111);
    std::vector<double> second = GenerateDenseMatrix(k_size_, k_size_, percent_second_, 222);
    correct_data_ = MultDense(static_cast<int>(k_size_), static_cast<int>(k_size_), first, static_cast<int>(k_size_),
                              static_cast<int>(k_size_), second);

    Ccs test_m1 = ConvertDense(k_size_, k_size_, first);
    Ccs test_m2 = ConvertDense(k_size_, k_size_, second);

    input_data_ = std::make_tuple(test_m1, test_m2);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rows = std::get<0>(output_data);
    int cols = std::get<1>(output_data);
    std::vector<double> task_res = std::get<2>(output_data);
    return CompareDenseResults(rows, cols, correct_data_, task_res);
  }
  static std::vector<double> GenerateDenseMatrix(size_t rows, size_t cols, double percent, int seed) {
    std::vector<double> dense(rows * cols, 0.0);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> r_dist;
    std::uniform_int_distribution<> row_dist(0, static_cast<int>(rows - 1));
    std::uniform_int_distribution<> col_dist(0, static_cast<int>(cols - 1));

    int el_count = static_cast<int>(static_cast<double>(rows * cols) * percent);
    int count_in_col = el_count / static_cast<int>(cols);
    el_count = count_in_col * static_cast<int>(cols);

    int created_el = 0;
    while (created_el < el_count) {
      auto row = static_cast<size_t>(row_dist(gen));
      auto col = static_cast<size_t>(col_dist(gen));

      if (dense[(cols * row) + col] == 0.0) {
        dense[(cols * row) + col] = r_dist(gen);
        created_el++;
      }
    }

    return dense;
  }

  static std::vector<double> MultDense(int rows1, int cols1, const std::vector<double> &m1, int rows2, int cols2,
                                       const std::vector<double> &m2) {
    if (rows2 != cols1) {
      throw std::runtime_error("cant multiplicate matrix");
    }
    std::vector<double> result(static_cast<size_t>(rows1) * static_cast<size_t>(cols2));

    for (int i = 0; i < rows1; i++) {
      for (int j = 0; j < cols2; j++) {
        for (int k = 0; k < cols1; k++) {
          result[(cols2 * i) + j] += m1[(cols1 * i) + k] * m2[(cols2 * k) + j];
        }
      }
    }

    return result;
  }

  static Ccs ConvertDense(size_t rows, size_t cols, const std::vector<double> &dense) {
    Ccs result;
    result.rows = static_cast<int>(rows);
    result.cols = static_cast<int>(cols);
    result.col_index.resize(cols + 1);

    for (size_t col = 0; col < cols; col++) {
      result.col_index[col] = static_cast<int>(result.val.size());
      for (size_t row = 0; row < rows; row++) {
        if (dense[(cols * row) + col] != 0) {
          result.val.push_back(dense[(cols * row) + col]);
          result.row.push_back(static_cast<int>(row));
        }
      }
    }
    result.col_index[cols] = static_cast<int>(result.val.size());
    return result;
  }

  static bool CompareDenseResults(int rows, int cols, const std::vector<double> &expected,
                                  const std::vector<double> &actual, double eps = 1e-5) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        int ind = (i * cols) + j;
        if (std::abs(expected[ind] - actual[ind]) > eps) {
          return false;
        }
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(KrapivinACcsMultPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, KrapivinACcsMultMPI, KrapivinACcsMultSEQ>(PPC_SETTINGS_krapivin_a_ccs_mult);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KrapivinACcsMultPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KrapivinACcsMultPerfTest, kGtestValues, kPerfTestName);

}  // namespace krapivin_a_ccs_mult
