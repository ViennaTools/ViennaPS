#ifndef PS_QUEUES_HPP
#define PS_QUEUES_HPP

/**
 * File: psQueues.hpp
 * Original Author: Keith Schwarz (htiek@cs.stanford.edu)
 * URL: https://www.keithschwarz.com/interesting/
 */

#include <functional>
#include <limits>
#include <map>
#include <utility>

// A bounded priority queue implementation.
// If a certain predifined number of elements are already stored in the queue,
// then a new item with a worse value than the worst already in the queue won't
// be added when enqueue is called with the new item.
template <class K, class V, typename Comparator = std::less<K>>
struct psBoundedPQueue {
  using MapType = std::multimap<K, V, Comparator>;
  using SizeType = std::size_t;

private:
  MapType mmap;
  const SizeType maximumSize;

public:
  psBoundedPQueue(SizeType passedMaximumSize)
      : maximumSize(passedMaximumSize) {}

  void enqueue(std::pair<K, V> &&item) {
    // Optimization: If this isn't going to be added, don't add it.
    if (size() == maxSize() && mmap.key_comp()(worst(), item.first))
      return;

    // Add the element to the collection.
    mmap.emplace(std::forward<std::pair<K, V>>(item));

    // If there are too many elements in the queue, drop off the last one.
    if (size() > maxSize()) {
      auto last = mmap.end();
      --last; // Now points to highest-priority element
      mmap.erase(last);
    }
  }

  V dequeueBest() {
    // Copy the best value.
    V result = mmap.begin()->second;

    // Remove it from the map.
    mmap.erase(mmap.begin());

    return result;
  }

  [[nodiscard]] SizeType maxSize() const { return maximumSize; }

  [[nodiscard]] SizeType size() const { return mmap.size(); }

  [[nodiscard]] bool empty() const { return mmap.empty(); }

  [[nodiscard]] K best() const {
    return mmap.empty() ? std::numeric_limits<K>::infinity()
                        : mmap.begin()->first;
  }

  [[nodiscard]] K worst() const {
    return mmap.empty() ? std::numeric_limits<K>::infinity()
                        : mmap.rbegin()->first;
  }
};

// A clamped priority queue implementation.
// Only items whose value is better than that of a predefined threshold can be
// added to the queue.
template <class K, class V, typename Comparator = std::less<K>>
struct psClampedPQueue {
  using MapType = std::multimap<K, V, Comparator>;
  using SizeType = std::size_t;

private:
  MapType mmap;
  const K thresValue;

public:
  psClampedPQueue(K passedThresValue) : thresValue(passedThresValue) {}

  void enqueue(std::pair<K, V> &&item) {
    // Optimization: If this isn't going to be added, don't add it.
    if (mmap.key_comp()(thresValue, item.first))
      return;

    // Add the element to the collection.
    mmap.emplace(std::forward<std::pair<K, V>>(item));
  }

  V dequeueBest() {
    // Copy the best value.
    V result = mmap.begin()->second;

    // Remove it from the map.
    mmap.erase(mmap.begin());

    return result;
  }

  [[nodiscard]] K thresholdValue() const { return thresValue; }

  [[nodiscard]] SizeType size() const { return mmap.size(); }

  [[nodiscard]] bool empty() const { return mmap.empty(); }

  [[nodiscard]] K best() const {
    return mmap.empty() ? std::numeric_limits<K>::infinity()
                        : mmap.begin()->first;
  }

  [[nodiscard]] K worst() const {
    return mmap.empty() ? std::numeric_limits<K>::infinity()
                        : mmap.rbegin()->first;
  }
};
#endif