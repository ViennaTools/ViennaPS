#pragma once

#include "psMaterial.hpp"

namespace viennaps {

template <class T> class MaterialValueMap {
public:
  using Value = T;

  MaterialValueMap() = default;

  // generic constructor (works with map, vector<pair>, initializer_list, etc.)
  template <class MapLike>
  explicit MaterialValueMap(const MapLike &mapLike, T defaultValue = T{})
      : default_(defaultValue) {
    for (const auto &[key, value] : mapLike) {
      set(key, value);
    }
  }

  static MaterialValueMap fromDefault(T defaultValue) {
    MaterialValueMap map;
    map.setDefault(defaultValue);
    return map;
  }

  void set(Material material, const T &value) {
    if (material.isBuiltIn()) {
      auto idx = toIndex(material.builtIn());
      values_[idx] = value;
      isSet_[idx] = true;
    } else {
      customValues_[material.customId()] = value;
    }
  }

  // perfect-forwarding overload
  template <class... Args> void emplace(BuiltInMaterial m, Args &&...args) {
    auto idx = toIndex(m);
    values_[idx] = T(std::forward<Args>(args)...);
    isSet_[idx] = true;
  }

  template <class... Args> void emplace(Material material, Args &&...args) {
    set(material, T(std::forward<Args>(args)...));
  }

  T &operator[](Material material) {
    if (material.isBuiltIn()) {
      auto idx = toIndex(material.builtIn());
      isSet_[idx] = true;
      return values_[idx];
    }
    return customValues_[material.customId()];
  }

  // get value or default T{}
  [[nodiscard]] T get(Material material) const {
    if (material.isBuiltIn()) {
      auto idx = toIndex(material.builtIn());
      return getBuiltInValue(idx);
    }

    const auto it = customValues_.find(material.customId());
    return it == customValues_.end() ? default_ : it->second;
  }

  [[nodiscard]] T get(BuiltInMaterial m) const {
    auto idx = toIndex(m);
    return getBuiltInValue(idx);
  }

  // get reference (no copy)
  [[nodiscard]] const T &getRef(Material material) const {
    if (material.isBuiltIn()) {
      auto idx = toIndex(material.builtIn());
      return getBuiltInValue(idx);
    }

    const auto it = customValues_.find(material.customId());
    return it == customValues_.end() ? default_ : it->second;
  }

  [[nodiscard]] const T &getRef(BuiltInMaterial m) const {
    auto idx = toIndex(m);
    return getBuiltInValue(idx);
  }

  const T getEntryByIndex(std::size_t idx) const {
    if (auto it = begin().goToIndex(idx); it != end()) {
      return (*it).value;
    } else {
      throw std::out_of_range("Index out of range in MaterialValueMap.");
    }
  }

  void setDefault(const T &v) { default_ = v; }

  [[nodiscard]] const T &getDefault() const { return default_; }

  // check if user provided a value
  [[nodiscard]] bool has(Material material) const {
    if (material.isBuiltIn()) {
      return isSet_[toIndex(material.builtIn())];
    }
    return customValues_.contains(material.customId());
  }

  [[nodiscard]] bool has(BuiltInMaterial m) const { return has(Material(m)); }

  bool empty() const {
    for (const auto &b : isSet_) {
      if (b) {
        return false;
      }
    }
    return customValues_.empty();
  }

  // remove value -> fallback to default
  void clear(Material material) {
    if (material.isBuiltIn()) {
      isSet_[toIndex(material.builtIn())] = false;
      return;
    }
    customValues_.erase(material.customId());
  }

  void clear(BuiltInMaterial m) { clear(Material(m)); }

  void clearAll() {
    for (auto &b : isSet_) {
      b = false;
    }
    customValues_.clear();
  }

  // ================= ITERATOR =================
  struct Entry {
    Material material;
    const T &value;
  };

  class Iterator {
  public:
    using CustomIterator =
        typename std::unordered_map<Material::ValueType, T>::const_iterator;

    Iterator(const MaterialValueMap *map, bool isEnd)
        : map_(map), phase_(isEnd ? Phase::End : Phase::BuiltIn), idx_(0) {
      if (!isEnd) {
        advanceToValid();
      }
    }

    Iterator goToIndex(std::size_t targetIdx) {
      idx_ = 0;
      phase_ = Phase::BuiltIn;
      advanceToValid();

      size_t idx = 0;
      while (phase_ != Phase::End && idx != targetIdx) {
        ++(*this);
        ++idx;
      }
      return *this;
    }

    Iterator &operator++() {
      if (phase_ == Phase::BuiltIn) {
        ++idx_;
        advanceToValid();
      } else if (phase_ == Phase::Custom) {
        ++customIt_;
        if (customIt_ == map_->customValues_.cend()) {
          phase_ = Phase::End;
        }
      }
      return *this;
    }

    bool operator!=(const Iterator &other) const {
      if (phase_ != other.phase_) {
        return true;
      }

      switch (phase_) {
      case Phase::BuiltIn:
        return idx_ != other.idx_;
      case Phase::Custom:
        return customIt_ != other.customIt_;
      case Phase::End:
        return false;
      }
      return false;
    }

    Entry operator*() const {
      if (phase_ == Phase::BuiltIn) {
        return Entry{Material(static_cast<BuiltInMaterial>(idx_)),
                     map_->values_[idx_]};
      }

      return Entry{Material::custom(customIt_->first), customIt_->second};
    }

  private:
    enum class Phase { BuiltIn, Custom, End };

    void advanceToValid() {
      while (idx_ < map_->values_.size() && !map_->isSet_[idx_]) {
        ++idx_;
      }

      if (idx_ >= map_->values_.size()) {
        phase_ = Phase::Custom;
        customIt_ = map_->customValues_.cbegin();
        if (customIt_ == map_->customValues_.cend()) {
          phase_ = Phase::End;
        }
      }
    }

    const MaterialValueMap *map_;
    Phase phase_;
    std::size_t idx_;
    CustomIterator customIt_{};
  };

  // iterators over set built-in materials and custom materials
  Iterator begin() const { return Iterator(this, false); }

  Iterator end() const { return Iterator(this, true); }

  const auto &getCustomMaterialValues() const { return customValues_; }

private:
  static constexpr std::size_t toIndex(BuiltInMaterial m) {
    return static_cast<std::uint16_t>(m);
  }

  const T &getBuiltInValue(std::size_t idx) const {
    return isSet_[idx] ? values_[idx] : default_;
  }

  std::array<T, kBuiltInMaterialMaxId + 1> values_{};
  std::array<bool, kBuiltInMaterialMaxId + 1> isSet_{};
  std::unordered_map<Material::ValueType, T> customValues_;

  // default instance used for reference return
  T default_{};
};

} // namespace viennaps