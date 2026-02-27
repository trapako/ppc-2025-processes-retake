#include "krapivin_a_ccs_mult/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "krapivin_a_ccs_mult/common/include/common.hpp"
#include "util/include/util.hpp"

namespace krapivin_a_ccs_mult {

KrapivinACcsMultSEQ::KrapivinACcsMultSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KrapivinACcsMultSEQ::ValidationImpl() {
  return (std::get<0>(GetInput()).val.size() > 0) && (std::get<1>(GetInput()).val.size() > 0);
}

bool KrapivinACcsMultSEQ::PreProcessingImpl() {
  return true;
}

bool KrapivinACcsMultSEQ::RunImpl() {
  ccs m1 = std::get<0>(GetInput());
  ccs m2 = std::get<1>(GetInput());

  std::vector<double> dense((m1.rows * m2.cols), 0.0);
  int cols_count = m2.cols;

  //std::cout << "m2 cols: " << cols_count << "\n";
  for(int i = 0; i < cols_count; i++) {
    int j1 = m2.col_index[i];
    int j2 = m2.col_index[i + 1];
    int col = i;

    for(int j = j1; j < j2; j++) {
      int row = m2.row[j];
      int k1 = m1.col_index[row];
      int k2 = m1.col_index[row + 1];
      
      //std::cout << "col:" <<col << " row: " << row << "\n";

      for(int k = k1; k < k2; k++) {
        //std::cout << "row" << m1.row[k] << " val " << m1.val[k] << "\n";  
        dense[(m1.row[k] * m2.cols) + col] += m1.val[k] * m2.val[j];
      }
      //std::cout << "----\n\n";
    }
  }
  GetOutput() = std::make_tuple(m1.rows, m2.cols, dense);
  return true;
}

bool KrapivinACcsMultSEQ::PostProcessingImpl() {
  return true;
}
}  // namespace krapivin_a_ccs_mult
