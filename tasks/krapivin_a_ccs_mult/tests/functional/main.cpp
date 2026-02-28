#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "krapivin_a_ccs_mult/common/include/common.hpp"
#include "krapivin_a_ccs_mult/mpi/include/ops_mpi.hpp"
#include "krapivin_a_ccs_mult/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace krapivin_a_ccs_mult {

class KrapivinACcsMultRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<5>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int rows1 = std::get<0>(params);
    int cols1 = std::get<1>(params);
    double percent1 = std::get<2>(params);

    int rows2 = cols1;
    int cols2 = std::get<3>(params);
    double percent2 = std::get<4>(params);

    std::vector<double> first = GenerateDenseMatrix(rows1, cols1, percent1, 111);
    std::vector<double> second = GenerateDenseMatrix(rows2, cols2, percent2, 222);
    correct_data_ = MultDense(rows1, cols1, first, rows2, cols2, second);

    Ccs test_m1 = ConvertDense(rows1, cols1, first);
    Ccs test_m2 = ConvertDense(rows2, cols2, second);

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

 private:
  InType input_data_;
  std::vector<double> correct_data_;
};

namespace {

TEST_P(KrapivinACcsMultRunFuncTestsProcesses, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 4> kTestParam = {
    std::make_tuple(10, 10, 0.1, 10, 0.2, "3"), std::make_tuple(100, 50, 0.1, 100, 0.1, "1"),
    std::make_tuple(20, 100, 0.2, 100, 0.2, "2"), std::make_tuple(1000, 1000, 0.01, 1000, 0.01, "4")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<KrapivinACcsMultMPI, InType>(kTestParam, PPC_SETTINGS_krapivin_a_ccs_mult),
                   ppc::util::AddFuncTask<KrapivinACcsMultSEQ, InType>(kTestParam, PPC_SETTINGS_krapivin_a_ccs_mult));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    KrapivinACcsMultRunFuncTestsProcesses::PrintFuncTestName<KrapivinACcsMultRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, KrapivinACcsMultRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace krapivin_a_ccs_mult
