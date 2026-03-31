#pragma once

#include "psMaterials.hpp"

namespace viennaps {

template <class T> class MaterialValueMap {
public:
  using Value = T;

  MaterialValueMap() = default;

  // generic constructor (works with map, vector<pair>, initializer_list, etc.)
  template <class MapLike> explicit MaterialValueMap(const MapLike &mapLike) {
    for (const auto &[key, value] : mapLike) {
      set(key, value);
    }
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
    BuiltInMaterial material;
    const T *value;
    bool set;

    [[nodiscard]] bool isSet() const { return set; }
    [[nodiscard]] const T &getValue() const { return *value; }
    [[nodiscard]] BuiltInMaterial getMaterial() const { return material; }
  };

  class Iterator {
  public:
    Iterator(const MaterialValueMap *map, std::size_t idx)
        : map_(map), idx_(idx) {
      advanceToValid();
    }

    Iterator &operator++() {
      ++idx_;
      advanceToValid();
      return *this;
    }

    bool operator!=(const Iterator &other) const { return idx_ != other.idx_; }

    Entry operator*() const {
      return Entry{static_cast<BuiltInMaterial>(idx_), &map_->values_[idx_],
                   map_->isSet_[idx_]};
    }

  private:
    void advanceToValid() {
      while (idx_ < map_->values_.size() && !map_->isSet_[idx_]) {
        ++idx_;
      }
    }

    const MaterialValueMap *map_;
    std::size_t idx_;
  };

  // iterators over built-in materials with user-provided values
  Iterator begin() const { return Iterator(this, 0); }

  Iterator end() const { return Iterator(this, values_.size()); }

  const auto &getCustomMaterialValues() const { return customValues_; }

  auto &customMaterialRates() { return customValues_; }

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