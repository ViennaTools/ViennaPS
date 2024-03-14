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

template <class T>
std::vector<std::vector<T>> generatePoints(unsigned N, unsigned D) {
  std::random_device rd;
  std::vector<std::vector<T>> data(N);
  for (auto &d : data)
    d.resize(D);

#pragma omp parallel default(none) shared(N, D, data, rd)
  {
    auto engine = std::default_random_engine(rd());
    std::uniform_real_distribution<> d{-10., 10.};
#pragma omp for
    for (unsigned i = 0; i < N; ++i) {
      std::vector<T> point;
      point.reserve(D);
      std::generate_n(std::back_inserter(point), D,
                      [&d, &engine]() { return d(engine); });
      data[i].swap(point);
    }
  }
  return data;
}

int main(int argc, char *argv[]) {
  using NumericType = double;
  static constexpr int D = 3;

  // The number of points in the tree
  unsigned N = 1'000'000;
  if (argc > 1) {
    int tmp = std::atoi(argv[1]);
    if (tmp > 0)
      N = static_cast<unsigned>(tmp);
  }

  // The number of points to query the tree with
  unsigned M = 100'000;
  if (argc > 2) {
    int tmp = std::atoi(argv[2]);
    if (tmp > 0)
      M = static_cast<unsigned>(tmp);
  }

  // The number repetitions
  unsigned repetitions = 1;
  if (argc > 3) {
    int tmp = std::atoi(argv[3]);
    if (tmp > 0)
      repetitions = static_cast<unsigned>(tmp);
  }

  // Training Point generation
  std::cout << "Generating Training Points...\n";
  auto points = generatePoints<NumericType>(N, D);

  // Testing points generation
  std::cout << "Generating Testing Points...\n";
  auto testPoints = generatePoints<NumericType>(M, D);

  {
    std::cout << "Growing Tree...\n";
    psSmartPointer<psKDTree<NumericType>> tree = nullptr;
    auto startTime = getTime();
    for (unsigned i = 0; i < repetitions; ++i) {
      tree = psSmartPointer<psKDTree<NumericType>>::New(points);
      tree->build();
    }
    auto endTime = getTime();
    std::cout << "Tree grew in " << (endTime - startTime) / repetitions
              << "s\n";

    // Nearest Neighbors
    std::cout << "Finding Nearest Neighbors...\n";
    startTime = getTime();
    for (unsigned i = 0; i < repetitions; ++i) {
      for (const auto &pt : testPoints) [[maybe_unused]]
        auto result = tree->findNearest(pt);
    }
    endTime = getTime();

    std::cout << M << " nearest neighbor queries completed in "
              << (endTime - startTime) / repetitions << "s\n";
  }
}
