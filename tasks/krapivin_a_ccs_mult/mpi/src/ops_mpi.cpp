#include "krapivin_a_ccs_mult/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <tuple>
#include <vector>

#include "krapivin_a_ccs_mult/common/include/common.hpp"

namespace krapivin_a_ccs_mult {

KrapivinACcsMultMPI::KrapivinACcsMultMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KrapivinACcsMultMPI::ValidationImpl() {
  auto input = GetInput();
  const auto &a = std::get<0>(input);
  const auto &b = std::get<1>(input);

  if (a.rows <= 0 || a.cols <= 0 || b.rows <= 0 || b.cols <= 0) {
    return false;
  }
  if (a.cols != b.rows) {
    return false;
  }
  if (a.col_index.size() != static_cast<size_t>(a.cols) + 1 || b.col_index.size() != static_cast<size_t>(b.cols) + 1) {
    return false;
  }
  return true;
}

bool KrapivinACcsMultMPI::PreProcessingImpl() {
  const Ccs &m1 = std::get<0>(GetInput());
  const Ccs &m2 = std::get<1>(GetInput());
  local_result_.assign(static_cast<size_t>(m1.rows) * static_cast<size_t>(m2.cols), 0.0);
  return true;
}

bool KrapivinACcsMultMPI::RunImpl() {
  int rank = 0;
  int mpi_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  Ccs m1;
  Ccs m2;

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
    m1.col_index.resize(static_cast<size_t>(m1.cols) + 1);
    m2.val.resize(static_cast<size_t>(m2_count));
    m2.row.resize(static_cast<size_t>(m2_count));
    m2.col_index.resize(static_cast<size_t>(m2.cols) + 1);
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
  const int start_col = (rank * step) + (rank < rem ? rank : rem);
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
        local_result_[(static_cast<size_t>(row_m1) * static_cast<size_t>(ncols)) + static_cast<size_t>(col)] +=
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
    local_result_.resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));
  }
  MPI_Bcast(local_result_.data(), rows * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GetOutput() = std::make_tuple(rows, cols, local_result_);
  return true;
}
}  // namespace krapivin_a_ccs_mult
