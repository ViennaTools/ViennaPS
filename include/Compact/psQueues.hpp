#ifndef PS_QUEUES_HPP
#define PS_QUEUES_HPP

/**
 * File: cmQueues.hpp
 * Original Author: Keith Schwarz (htiek@cs.stanford.edu)
 * URL: https://www.keithschwarz.com/interesting/
 */

#include <functional>
#include <limits>
#include <map>
#include <utility>

template <class K, class V, typename Comparator = std::less<K>>
struct cmBoundedPQueue {
  using MapType = std::multimap<K, V, Comparator>;
  using SizeType = std::size_t;

private:
  MapType mmap;
  const SizeType maximumSize;

public:
  cmBoundedPQueue(SizeType passedMaximumSize)
      : maximumSize(passedMaximumSize) {}

  void enqueue(std::pair<K, V> &&item) {
    /* Optimization: If this isn't going to be added, don't add it. */
    if (size() == maxSize() && mmap.key_comp()(worst(), item.first))
      return;

    /* Add the element to the collection. */
    mmap.emplace(std::forward<std::pair<K, V>>(item));

    /* If there are too many elements in the queue, drop off the last one. */
    if (size() > maxSize()) {
      auto last = mmap.end();
      --last; // Now points to highest-priority element
      mmap.erase(last);
    }
  }

  V dequeueBest() {
    /* Copy the best value. */
    V result = mmap.begin()->second;

    /* Remove it from the map. */
    mmap.erase(mmap.begin());

    return result;
  }

  SizeType maxSize() const { return maximumSize; }

  SizeType size() const { return mmap.size(); }

  bool empty() const { return mmap.empty(); }

  K best() const {
    return mmap.empty() ? std::numeric_limits<K>::infinity()
                        : mmap.begin()->first;
  }

  K worst() const {
    return mmap.empty() ? std::numeric_limits<K>::infinity()
                        : mmap.rbegin()->first;
  }
};

template <class K, class V, typename Comparator = std::less<K>>
struct cmClampedPQueue {
  using MapType = std::multimap<K, V, Comparator>;
  using SizeType = std::size_t;

private:
  MapType mmap;
  const K thresValue;

public:
  cmClampedPQueue(K passedThresValue) : thresValue(passedThresValue) {}

  void enqueue(std::pair<K, V> &&item) {
    /* Optimization: If this isn't going to be added, don't add it. */
    if (mmap.key_comp()(thresValue, item.first))
      return;

    /* Add the element to the collection. */
    mmap.emplace(std::forward<std::pair<K, V>>(item));
  }

  V dequeueBest() {
    /* Copy the best value. */
    V result = mmap.begin()->second;

    /* Remove it from the map. */
    mmap.erase(mmap.begin());

    return result;
  }

  K thresholdValue() const { return thresValue; }

  SizeType size() const { return mmap.size(); }

  bool empty() const { return mmap.empty(); }

  K best() const {
    return mmap.empty() ? std::numeric_limits<K>::infinity()
                        : mmap.begin()->first;
  }

  K worst() const {
    return mmap.empty() ? std::numeric_limits<K>::infinity()
                        : mmap.rbegin()->first;
  }
};
#endif