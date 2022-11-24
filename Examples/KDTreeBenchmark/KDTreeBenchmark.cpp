#include <array>
#include <iostream>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <psKDTree.hpp>
#include <psSmartPointer.hpp>

inline double getTime() {
#ifdef _OPENMP
  return omp_get_wtime();
#else
  return std::chrono::duration<double>(
             std::chrono::high_resolution_clock::now().time_since_epoch())
      .count();
#endif
}

template <class T, int D> std::vector<std::array<T, D>> generatePoints(int N) {
  std::random_device rd;
  std::vector<std::array<T, D>> data(N);

#pragma omp parallel default(none) shared(N, data, rd)
  {
    int numThreads = 1;
#ifdef _OPENMP
    numThreads = omp_get_num_threads();
#endif

    auto engine = std::default_random_engine(rd());
    std::uniform_real_distribution<> d{-10., 10.};
#pragma omp for
    for (int i = 0; i < N; ++i) {
      if constexpr (D == 3) {
        data[i] = std::array<T, D>{d(engine), d(engine), d(engine)};
      } else {
        data[i] = std::array<T, D>{d(engine), d(engine)};
      }
    }
  }

  return data;
}

int main(int argc, char *argv[]) {
  using NumericType = double;
  static constexpr int D = 3;

  // The number of points in the tree
  int N = 1'000'000;
  if (argc > 1) {
    int tmp = std::atoi(argv[1]);
    if (tmp > 0)
      N = tmp;
  }

  // The number of points to query the tree with
  int M = 100'000;
  if (argc > 2) {
    int tmp = std::atoi(argv[2]);
    if (tmp > 0)
      M = tmp;
  }

  // The number repetitions
  int repetitions = 1;
  if (argc > 3) {
    int tmp = std::atoi(argv[3]);
    if (tmp > 0)
      repetitions = tmp;
  }

  // Training Point generation
  std::cout << "Generating Training Points...\n";
  auto points = generatePoints<NumericType, D>(N);

  // Testing points generation
  std::cout << "Generating Testing Points...\n";
  auto testPoints = generatePoints<NumericType, D>(M);

  {
    // Custom Tree
    std::cout << "Growing Tree...\n";
    double totalTime{0.};
    psSmartPointer<psKDTree<NumericType, D>> tree = nullptr;
    auto startTime = getTime();
    for (unsigned i = 0; i < repetitions; ++i) {
      tree = psSmartPointer<psKDTree<NumericType, D>>::New(points);
      tree->build();
    }
    auto endTime = getTime();
    std::cout << "Tree grew in " << (endTime - startTime) / repetitions
              << "s\n";

    totalTime = 0.;
    // Nearest Neighbors
    std::cout << "Finding Nearest Neighbors...\n";
    startTime = getTime();
    for (unsigned i = 0; i < repetitions; ++i) {
      for (const auto pt : testPoints)
        auto result = tree->findNearest(pt);
    }
    endTime = getTime();

    std::cout << M << " nearest neighbor queries completed in "
              << (endTime - startTime) / repetitions << "s\n";
  }
}
