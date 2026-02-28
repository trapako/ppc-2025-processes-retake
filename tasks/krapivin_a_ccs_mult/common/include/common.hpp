#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace krapivin_a_ccs_mult {

struct Ccs {
  int rows{};
  int cols{};

  std::vector<double> val;
  std::vector<int> row;
  std::vector<int> col_index;
};

using InType = std::tuple<Ccs, Ccs>;
using OutType = std::tuple<int, int, std::vector<double>>;
using TestType = std::tuple<int, int, double, int, double, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace krapivin_a_ccs_mult
