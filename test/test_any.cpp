#include "test_common.h"
//#include <parlay_hash/unordered_set.h>
#include <parlay/hash_table.h>

// function to generate a vector of random integers
std::vector<int> generate_random_vector(size_t size) {
  std::vector<int> v(size);
  for (size_t i = 0; i < size; i++) {
      v[i] = rand() % (2 * size);
  }
  return v;
}

int main() {
    int N = 10;
    auto vec1 = generate_random_vector(N);
    auto vec2 = generate_random_vector(N);
    fmt::print("vec1 = {}\n", vec1);
    fmt::print("vec2 = {}\n", vec2);

    parlay::hashtable<parlay::hash_numeric<int>> table(5 * N, parlay::hash_numeric<int>());

    parlay::parallel_for(0, 2 * N, [&](size_t i) {
        if (i < N)  table.insert(vec1[i]);
        else  table.insert(vec2[i - N]);
    });

    fmt::print("set = {}\n", table.entries());

    return 0;
}