#pragma once

#include <vector>

#include "task/include/task.hpp"

namespace krapivin_a_ring {

struct RingTaskData {
  int data;
  int source_rank;
  int target_rank;

  RingTaskData() = default;
  RingTaskData(int d, int s, int t) : data(d), source_rank(s), target_rank(t) {}
};

using InType = RingTaskData;
using OutType = std::vector<int>;
using TestType = int;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace krapivin_a_ring
