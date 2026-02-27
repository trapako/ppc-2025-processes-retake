#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <limits>
#include <random>
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
    return std::to_string(std::get<0>(test_param));
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int rows1 = std::get<0>(params);
    int cols1 = std::get<1>(params);
    double percent1 = std::get<2>(params);

    int rows2 = cols1;
    int cols2 = cols1 = std::get<3>(params);
    double percent2 = std::get<4>(params);

    ccs test_m1 = GenerateSparseMatrix(rows1, cols1, percent1, 111);
    ccs test_m2 = GenerateSparseMatrix(rows2, cols2, percent2, 222);

    //PrintCCS(test_m1);
    //PrintCCS(test_m2);
    correct_data_ = MultDense(rows1, cols1, ConvertCCS(test_m1), rows2, cols2, ConvertCCS(test_m2));
    //PrintDense(rows1, cols2, correct_data_);

    input_data_ = std::make_tuple(test_m1, test_m2);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rows = std::get<0>(output_data);
    int cols = std::get<1>(output_data);
    std::vector<double> task_res = std::get<2>(output_data);

    //PrintDense(rows, cols, correct_data_);
    PrintDense(rows, cols, task_res);

    double eps = 1e-5;
    for(int i = 0; i < rows; i++) {
      for(int j = 0; j < cols; j++) {
        int ind = (i * cols) + j;
        double sub = std::abs(correct_data_[ind] - task_res[ind]);
        if(sub > eps) {
          return false;
        }
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  ccs GenerateSparseMatrix(int rows, int cols, double percent,int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> r_dist;
    std::uniform_int_distribution<> row_dist(0, (rows - 1));
    std::uniform_int_distribution<> col_dist(0, (cols -1));

    int el_count = static_cast<int>(static_cast<double>(rows * cols) * percent);
    int count_in_col = el_count / cols;
    el_count = count_in_col * cols;

    std::cout << "el_count: " << el_count << "\n count_in_col: " << count_in_col << "\n";

    ccs matrix;
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.val.resize(count_in_col * cols, 0);
    matrix.row.resize(count_in_col * cols, 0);
    matrix.col_index.resize(cols + 1, 0);

    for(int i = 0; i < cols; i++) {
      for(int j = 0; j < count_in_col; j++) {
        bool flag = false;
        int gen_index = i * count_in_col + j;

        do {

          matrix.row[gen_index] = row_dist(gen);
          flag = false;

          for(int k = 0; k < j; k++) {
            int check_index = i * count_in_col + k;
            if(matrix.row[gen_index] == matrix.row[check_index]) {
              flag = true;
              break;
            }
          }

        } while(flag);

        std::sort(matrix.row.data() + (count_in_col * i), matrix.row.data() + (count_in_col * (i + 1)));
      }
    }

    for(int i = 0 ; i < el_count; i++) {
      matrix.val[i] = static_cast<double>(col_dist(gen));//r_dist(gen);
    }

    for(int i = 0; i < (cols+1); i ++) {
      matrix.col_index[i] = i * count_in_col;
    }
    return matrix;
  }

  std::vector<double> ConvertCCS(ccs matrix) {
    std::vector<double> dense(matrix.rows * matrix.cols, 0.0);
    for(int i = 0; i < matrix.cols; i ++) { 
      int j1 = matrix.col_index[i];
      int j2 = matrix.col_index[i + 1];
      for(int j = j1; j < j2; j++) {
        dense[matrix.cols * matrix.row[j] + i] = matrix.val[j];
      }
    }

    //PrintDense(matrix.rows, matrix.cols, dense);
    return dense; 
  }

  std::vector<double> MultDense(int rows1, int cols1, const std::vector<double>& m1, 
    int rows2, int cols2, const std::vector<double>& m2) {
    if (rows2 != cols1) {
      throw std::runtime_error("cant multiplicate matrix");
    }
    std::vector<double> result(rows1 * cols2);

    for(int i = 0; i < rows1; i++) {
      for(int j = 0; j < cols2; j++) {
        for(int k = 0; k < cols1; k++) {
          result[(cols2 * i) + j] += m1[(cols1 * i) + k] * m2[(cols2 * k) + j];
        }
      }
    }
    
    return result;
  }

  void PrintCCS(const ccs& m) {
    std::cout << "val : ";
    for(size_t i = 0; i < m.val.size(); i++) {
      std::cout << m.val[i] << " ";
    } 
    std::cout << "\nrow: ";
    for(size_t i = 0; i < m.row.size(); i++) {
      std::cout << m.row[i] << " ";
    }
    std::cout << "\ncol_index: ";
    for(int i = 0; i <= m.cols; i++)  {
      std::cout << m.col_index[i] << " ";
    }
    std::cout << "\n";
  }

  void PrintDense(int rows, int cols,std::vector<double> m) {
    std::cout << "\n";
    for(int i = 0; i < rows; i++) {
      for(int j = 0; j < cols; j++) {
        std::cout << m[(i*cols) + j] << " ";
      }
      std::cout << "\n";
    }
  }
 private:
  InType input_data_;
  std::vector<double> correct_data_;
};

namespace {

TEST_P(KrapivinACcsMultRunFuncTestsProcesses, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 1> kTestParam = {std::make_tuple(10, 10, 0.1, 10, 0.2, "3")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<KrapivinACcsMultMPI, InType>(kTestParam, PPC_SETTINGS_krapivin_a_ccs_mult),
    ppc::util::AddFuncTask<KrapivinACcsMultSEQ, InType>(kTestParam, PPC_SETTINGS_krapivin_a_ccs_mult));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    KrapivinACcsMultRunFuncTestsProcesses::PrintFuncTestName<KrapivinACcsMultRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, KrapivinACcsMultRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace krapivin_a_ccs_mult
