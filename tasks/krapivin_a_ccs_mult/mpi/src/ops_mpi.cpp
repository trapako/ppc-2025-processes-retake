#include "krapivin_a_ccs_mult/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <vector>

#include "krapivin_a_ccs_mult/common/include/common.hpp"
#include "util/include/util.hpp"

namespace krapivin_a_ccs_mult {

KrapivinACcsMultMPI::KrapivinACcsMultMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KrapivinACcsMultMPI::ValidationImpl() {
  auto input = GetInput();
  const auto &A = std::get<0>(input);
  const auto &B = std::get<1>(input);

  if (A.rows <= 0 || A.cols <= 0 || B.rows <= 0 || B.cols <= 0) {
    return false;
  }
  if (A.cols != B.rows) {
    return false;
  }
  if (A.col_index.size() != static_cast<size_t>(A.cols + 1) || B.col_index.size() != static_cast<size_t>(B.cols + 1)) {
    return false;
  }
  return true;
}

bool KrapivinACcsMultMPI::PreProcessingImpl() {
  const ccs &m1 = std::get<0>(GetInput());
  const ccs &m2 = std::get<1>(GetInput());
  local_result_.assign(static_cast<size_t>(m1.rows * m2.cols), 0.0);
  return true;
}

bool KrapivinACcsMultMPI::RunImpl() {
  int rank = 0;
  int mpi_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  ccs m1;
  ccs m2;

  if (rank == 0) {
    m1 = std::get<0>(GetInput());
    m2 = std::get<1>(GetInput());
  }

  MPI_Bcast(&m1.rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&m1.cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&m2.rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&m2.cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  const int m1_el_count = (rank == 0) ? static_cast<int>(m1.val.size()) : 0;
  const int m2_el_count = (rank == 0) ? static_cast<int>(m2.val.size()) : 0;
  int m1_count = m1_el_count;
  int m2_count = m2_el_count;
  MPI_Bcast(&m1_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&m2_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    m1.val.resize(static_cast<size_t>(m1_count));
    m1.row.resize(static_cast<size_t>(m1_count));
    m1.col_index.resize(static_cast<size_t>(m1.cols + 1));
    m2.val.resize(static_cast<size_t>(m2_count));
    m2.row.resize(static_cast<size_t>(m2_count));
    m2.col_index.resize(static_cast<size_t>(m2.cols + 1));
  }

  MPI_Bcast(m1.val.data(), m1_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(m1.row.data(), m1_count, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(m1.col_index.data(), m1.cols + 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(m2.val.data(), m2_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(m2.row.data(), m2_count, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(m2.col_index.data(), m2.cols + 1, MPI_INT, 0, MPI_COMM_WORLD);

  const int ncols = m2.cols;
  const int step = ncols / mpi_size;
  const int rem = ncols % mpi_size;
  const int start_col = rank * step + (rank < rem ? rank : rem);
  const int my_col_count = step + (rank < rem ? 1 : 0);

  for (int col = start_col; col < start_col + my_col_count; ++col) {
    const int j1 = m2.col_index[col];
    const int j2 = m2.col_index[col + 1];
    for (int j = j1; j < j2; ++j) {
      const int row_m2 = m2.row[j];
      const int k1 = m1.col_index[row_m2];
      const int k2 = m1.col_index[row_m2 + 1];
      for (int k = k1; k < k2; ++k) {
        const int row_m1 = m1.row[k];
        local_result_[static_cast<size_t>(row_m1 * ncols + col)] +=
            m1.val[static_cast<size_t>(k)] * m2.val[static_cast<size_t>(j)];
      }
    }
  }

  const int result_count = m1.rows * m2.cols;

  if (rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, local_result_.data(), result_count, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    GetOutput() = std::make_tuple(m1.rows, m2.cols, local_result_);
  } else {
    MPI_Reduce(local_result_.data(), nullptr, result_count, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  }
  return true;
}

bool KrapivinACcsMultMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int rows = 0;
  int cols = 0;
  if (rank == 0) {
    rows = std::get<0>(GetOutput());
    cols = std::get<1>(GetOutput());
  }
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    local_result_.resize(static_cast<size_t>(rows * cols));
  }
  MPI_Bcast(local_result_.data(), rows * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GetOutput() = std::make_tuple(rows, cols, local_result_);
  return true;
}

void KrapivinACcsMultMPI::PrintCCS(const ccs &m) {
  std::cout << "val : ";
  for (size_t i = 0; i < m.val.size(); i++) {
    std::cout << m.val[i] << " ";
  }
  std::cout << "\nrow: ";
  for (size_t i = 0; i < m.row.size(); i++) {
    std::cout << m.row[i] << " ";
  }
  std::cout << "\ncol_index: ";
  for (int i = 0; i <= m.cols; i++) {
    std::cout << m.col_index[i] << " ";
  }
  std::cout << "\n";
}

}  // namespace krapivin_a_ccs_mult
