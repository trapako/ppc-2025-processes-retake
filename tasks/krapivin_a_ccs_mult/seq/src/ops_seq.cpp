#include "krapivin_a_ccs_mult/seq/include/ops_seq.hpp"

#include <cstddef>
#include <tuple>
#include <vector>

#include "krapivin_a_ccs_mult/common/include/common.hpp"

namespace krapivin_a_ccs_mult {

KrapivinACcsMultSEQ::KrapivinACcsMultSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KrapivinACcsMultSEQ::ValidationImpl() {
  const auto &m1 = std::get<0>(GetInput());
  const auto &m2 = std::get<1>(GetInput());
  if (m1.val.empty() || m2.val.empty()) {
    return false;
  }

  if (m1.cols != m2.rows) {
    return false;
  }
  if (m1.col_index.size() != static_cast<size_t>(m1.cols) + 1 ||
      m2.col_index.size() != static_cast<size_t>(m2.cols) + 1) {
    return false;
  }
  return true;
}

bool KrapivinACcsMultSEQ::PreProcessingImpl() {
  return true;
}

bool KrapivinACcsMultSEQ::RunImpl() {
  Ccs m1 = std::get<0>(GetInput());
  Ccs m2 = std::get<1>(GetInput());

  int result_rows = m1.rows;
  int result_cols = m2.cols;
  std::vector<double> dense(static_cast<size_t>(result_rows) * static_cast<size_t>(result_cols), 0.0);

  for (int col_m2 = 0; col_m2 < m2.cols; col_m2++) {
    int j_start = m2.col_index[col_m2];
    int j_end = m2.col_index[col_m2 + 1];

    for (int j = j_start; j < j_end; j++) {
      int row_m2 = m2.row[j];

      int k_start = m1.col_index[row_m2];
      int k_end = m1.col_index[row_m2 + 1];

      for (int k = k_start; k < k_end; k++) {
        int row_result = m1.row[k];

        dense[(row_result * result_cols) + col_m2] += m1.val[k] * m2.val[j];
      }
    }
  }

  GetOutput() = std::make_tuple(result_rows, result_cols, dense);
  return true;
}

bool KrapivinACcsMultSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace krapivin_a_ccs_mult
