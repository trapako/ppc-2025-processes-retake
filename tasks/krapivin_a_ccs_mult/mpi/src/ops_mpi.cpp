// ops_mpi.cpp
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
  // Проверяем, что входные данные корректны
  auto input = GetInput();
  const auto &A = std::get<0>(input);
  const auto &B = std::get<1>(input);

  // Размерности должны быть положительными
  if (A.rows <= 0 || A.cols <= 0 || B.rows <= 0 || B.cols <= 0) {
    return false;
  }
  // Умножение возможно, если число столбцов A равно числу строк B
  if (A.cols != B.rows) {
    return false;
  }
  // Проверка целостности структур CCS (опционально)
  if (A.col_index.size() != static_cast<size_t>(A.cols + 1) ||
      B.col_index.size() != static_cast<size_t>(B.cols + 1)) {
    return false;
  }
  return true;
}

bool KrapivinACcsMultMPI::PreProcessingImpl() {
  // Извлекаем входные матрицы
  ccs m1 = std::get<0>(GetInput());
  ccs m2 = std::get<1>(GetInput());

  local_result_.resize(m1.rows * m2.cols, 0.0);

  return true;
}

bool KrapivinACcsMultMPI::RunImpl() {
  int rank, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  
  ccs m1, m2;
  int m1_el_count = 0, m2_el_count = 0;
  std::vector<double> global_result;
  if(rank == 0) {
    m1 = std::get<0>(GetInput());
    m2 = std::get<1>(GetInput());
    m1_el_count = static_cast<int>(m1.val.size());
    m2_el_count = static_cast<int>(m2.val.size());
    global_result.resize(m1.rows * m2.cols, 0.0);
  }
  MPI_Bcast(&m1.rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&m1.cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&m1_el_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if(rank != 0) {
    m1.val.resize(m1_el_count);
    m1.row.resize(m1_el_count);
    m1.col_index.resize(m1.cols + 1);
  }
  MPI_Bcast(m1.val.data(), static_cast<int>(m1.val.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(m1.row.data(), static_cast<int>(m1.row.size()), MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(m1.col_index.data(), static_cast<int>(m1.col_index.size()), MPI_INT, 0, MPI_COMM_WORLD);
  
  std::cout << "rank: " << rank << " m1 good\n";

  MPI_Bcast(&m2.rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&m2.cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&m2_el_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if(rank != 0) {
    m2.val.resize(m2_el_count);
    m2.row.resize(m2_el_count);
    m2.col_index.resize(m2.cols + 1);
  }
  MPI_Bcast(m2.val.data(), static_cast<int>(m2.val.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(m2.row.data(), static_cast<int>(m2.row.size()), MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(m2.col_index.data(), static_cast<int>(m2.col_index.size()), MPI_INT, 0, MPI_COMM_WORLD);
  
  // if(rank ==2) {
  //   PrintCCS(m1);
    
  //   PrintCCS(m2);
  // }

  //std::cout << "rank: " << rank << " m2 good\n";
  
  int step = m2.cols / mpi_size;
  int rem = m2.cols % mpi_size;
  std::vector<int> col_per_proc(mpi_size, step);
  for(int i = 0; i < rem; i++){
    col_per_proc[i]++;
  }

  int start_col = 0;

  for(int i = 0; i < rank; i++) {
    start_col += col_per_proc[i];
  }

  //std::cout << "rank : " << rank << " col per proc "<< col_per_proc[rank] <<" start_col " << start_col <<"\n";

  // Локальное умножение для назначенных столбцов
  for (int i = start_col; i < start_col + col_per_proc[rank]; i++) {
    int col = i;
    int j1 = m2.col_index[col];
    int j2 = m2.col_index[col + 1];
    
    // if(rank == 0) {
    //     std::cout << j1 << " " << j2 << "\n";
    //   }

    for (int j = j1; j < j2; j++) {
      int row_m2 = m2.row[j];
      int k1 = m1.col_index[row_m2];
      int k2 = m1.col_index[row_m2 + 1];

      // if(rank == 0) {
      //   std::cout << k1 << " " << k2 << "\n";
      // }

      for (int k = k1; k < k2; ++k) {
        int row_m1 = m1.row[k];
        // if(rank == 0) {
        //   std::cout << m1.val[k] * m2.val[j] << " ";
        // }
        local_result_[(row_m1 * m2.cols) + i] += m1.val[k] * m2.val[j];
      }
      // if(rank == 0) {
      //     std::cout << "\n";
      // }
    }
  }


  // std::cout << std::endl;
  //   for(int i = 0; i < m1.rows; i++) {
  //     for(int j = 0; j < m2.cols; j++) {
  //       std::cout << local_result_[(i*m2.cols) + j] << " ";
  //     }
  //     std::cout << "\n";
  //   }
  MPI_Reduce(local_result_.data(), global_result.data(), static_cast<int>(local_result_.size()), 
      MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if(rank == 0) {
    GetOutput() = std::make_tuple(m1.rows, m2.cols, global_result);
  }
  
  return true;
}

bool KrapivinACcsMultMPI::PostProcessingImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rows = 0, cols = 0;
  if(rank == 0) {
    local_result_ = std::get<2>(GetOutput());
    rows = std::get<0>(GetOutput());
    cols = std::get<1>(GetOutput());
  }
  MPI_Bcast(local_result_.data(), static_cast<int>(local_result_.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GetOutput() = std::make_tuple(rows, cols, local_result_);

  return true;
}

void KrapivinACcsMultMPI::PrintCCS(const ccs& m) {
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

}  // namespace krapivin_a_ccs_mult