#pragma once

#include <unordered_map>

template <class T> class csTracePath {
private:
  std::unordered_map<int, T> data;
  std::vector<T> gridData;

public:
  std::unordered_map<int, T> &getData() { return data; }

  std::vector<T> &getGridData() { return gridData; }

  T getGridValue(int idx) const { return gridData[idx]; }

  void addPoint(int idx, T value) {
    auto search = data.find(idx);
    if (search != data.end()) {
      data[idx] += value;
    } else {
      data.insert({idx, value});
    }
  }

  void addPath(csTracePath &path) {
    for (const auto it : path.getData()) {
      addPoint(it.first, it.second);
    }
  }

  void useGridData(size_t numCells) { gridData.resize(numCells, 0.); }

  void addGridData(int idx, T value) { gridData[idx] += value; }

  void clear() {
    data.clear();
    gridData.clear();
  }
};