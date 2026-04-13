#pragma once

#include <lsMesh.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace viennaps {

using namespace viennacore;

/// Statistics collected during constraint cleaning
struct ConstraintCleanerStats {
  // Input stats
  size_t inputPoints = 0;
  size_t inputEdges = 0;

  // After normalization
  size_t normalizedPoints = 0;
  size_t normalizedEdges = 0;
  size_t removedDuplicateEdges = 0;
  size_t removedInvalidEdges = 0;

  // Graph structure
  size_t numComponents = 0;
  size_t numJunctions = 0; // deg > 2
  size_t numEndpoints = 0; // deg == 1
  size_t numBranches = 0;
  size_t numCycles = 0;

  // Edge length stats
  double minEdgeLength = 0;
  double maxEdgeLength = 0;
  double medianEdgeLength = 0;
  size_t edgesBelowLMin = 0;

  // Operations performed
  size_t mergedVertices = 0;
  size_t collapsedEdges = 0;
  size_t insertedPoints = 0;

  // Output stats
  size_t outputPoints = 0;
  size_t outputEdges = 0;

  void print() const {
    Logger::getInstance()
        .addInfo("ConstraintCleaner Statistics:")
        .addInfo("  Input: " + std::to_string(inputPoints) + " points, " +
                 std::to_string(inputEdges) + " edges")
        .addInfo("  After normalization: " + std::to_string(normalizedPoints) +
                 " points, " + std::to_string(normalizedEdges) + " edges")
        .addInfo("    Removed duplicate edges: " +
                 std::to_string(removedDuplicateEdges))
        .addInfo("    Removed invalid edges: " +
                 std::to_string(removedInvalidEdges))
        .addInfo("  Graph structure:")
        .addInfo("    Components: " + std::to_string(numComponents))
        .addInfo("    Junctions (deg>2): " + std::to_string(numJunctions))
        .addInfo("    Endpoints (deg==1): " + std::to_string(numEndpoints))
        .addInfo("    Branches: " + std::to_string(numBranches))
        .addInfo("    Cycles: " + std::to_string(numCycles))
        .addInfo("  Edge lengths: min=" + std::to_string(minEdgeLength) +
                 ", median=" + std::to_string(medianEdgeLength) +
                 ", max=" + std::to_string(maxEdgeLength))
        .addInfo("    Edges below l_min: " + std::to_string(edgesBelowLMin))
        .addInfo("  Operations:")
        .addInfo("    Merged vertices: " + std::to_string(mergedVertices))
        .addInfo("    Collapsed edges: " + std::to_string(collapsedEdges))
        .addInfo("    Inserted points: " + std::to_string(insertedPoints))
        .addInfo("  Output: " + std::to_string(outputPoints) + " points, " +
                 std::to_string(outputEdges) + " edges")
        .print();
  }
};

/// A cleaned constraint set suitable for CGAL CDT
template <typename NumericType> struct CleanedConstraints {
  std::vector<Vec2D<NumericType>> points;
  std::vector<std::array<unsigned, 2>> edges;
  std::vector<std::vector<unsigned>> polylines; // ordered vertex sequences
};

/**
 * @brief Cleans and preprocesses constraint edges for CGAL CDT.
 *
 * This class takes a 2D line mesh (points + edges) and produces a cleaned
 * constraint set suitable for CGAL Constrained Delaunay Triangulation:
 * - Junction vertices (degree > 2) are preserved
 * - Near-duplicate points are merged
 * - Tiny segments are collapsed
 * - Optional resampling for uniform edge lengths
 */
template <typename NumericType> class ConstraintCleaner {
public:
  using Point2D = Vec2D<NumericType>;
  using Edge = std::array<unsigned, 2>;

private:
  // Input
  std::vector<Point2D> inputPoints_;
  std::vector<Edge> inputEdges_;

  // Parameters
  NumericType hTarget_ = -1;     // target edge spacing (auto-computed if < 0)
  NumericType epsMerge_ = -1;    // merge threshold (auto-computed if < 0)
  NumericType lMin_ = -1;        // minimum edge length (auto-computed if < 0)
  NumericType simplifyTol_ = -1; // simplification tolerance (auto-computed)
  NumericType angleThreshold_ = 30.0; // degrees, for sharp corner detection
  bool enableSimplification_ = false;
  bool enableResampling_ = true;
  bool verbose_ = false;

  // Internal graph representation
  struct Graph {
    std::vector<Point2D> points;
    std::vector<Edge> edges;
    std::vector<std::vector<unsigned>> adj; // adjacency lists (neighbor vids)
    std::vector<unsigned> degree;
    std::vector<bool> isProtected; // junction, endpoint, or user-marked
    std::vector<bool> isDeleted;   // soft-deleted vertices

    void clear() {
      points.clear();
      edges.clear();
      adj.clear();
      degree.clear();
      isProtected.clear();
      isDeleted.clear();
    }

    void resize(size_t n) {
      adj.resize(n);
      degree.resize(n, 0);
      isProtected.resize(n, false);
      isDeleted.resize(n, false);
    }
  };

  Graph graph_;

  // Branch representation
  struct Branch {
    std::vector<unsigned> vertices; // ordered vertex indices
    bool isCycle = false;
  };
  std::vector<Branch> branches_;

  // Output
  CleanedConstraints<NumericType> output_;
  ConstraintCleanerStats stats_;

  // Helper: compute squared distance between two points
  NumericType distanceSquared(const Point2D &a, const Point2D &b) const {
    NumericType dx = a[0] - b[0];
    NumericType dy = a[1] - b[1];
    return dx * dx + dy * dy;
  }

  NumericType distance(const Point2D &a, const Point2D &b) const {
    return std::sqrt(distanceSquared(a, b));
  }

  // Helper: canonicalize edge (u < v)
  Edge canonicalize(unsigned u, unsigned v) const {
    return (u < v) ? Edge{u, v} : Edge{v, u};
  }

  // Stage 1: Validate and normalize input
  void normalizeInput() {
    graph_.clear();
    graph_.points = inputPoints_;
    graph_.resize(inputPoints_.size());

    std::set<std::pair<unsigned, unsigned>> uniqueEdges;
    size_t invalidCount = 0;

    for (const auto &e : inputEdges_) {
      unsigned u = e[0], v = e[1];

      // Check validity
      if (u >= inputPoints_.size() || v >= inputPoints_.size()) {
        ++invalidCount;
        continue;
      }
      if (u == v) { // self-loop
        ++invalidCount;
        continue;
      }

      // Canonicalize and check for duplicates
      auto canonical = std::make_pair(std::min(u, v), std::max(u, v));
      if (uniqueEdges.count(canonical)) {
        ++stats_.removedDuplicateEdges;
        continue;
      }
      uniqueEdges.insert(canonical);

      graph_.edges.push_back(canonicalize(u, v));
      graph_.adj[u].push_back(v);
      graph_.adj[v].push_back(u);
      ++graph_.degree[u];
      ++graph_.degree[v];
    }

    stats_.removedInvalidEdges = invalidCount;
    stats_.normalizedPoints = graph_.points.size();
    stats_.normalizedEdges = graph_.edges.size();
  }

  // Mark protected vertices (junctions, endpoints)
  void markProtectedVertices() {
    for (size_t v = 0; v < graph_.points.size(); ++v) {
      if (graph_.isDeleted[v])
        continue;
      unsigned deg = graph_.degree[v];
      // Protect junctions (deg > 2) and endpoints (deg == 1)
      // Also protect isolated vertices (deg == 0) to avoid removing them
      if (deg != 2) {
        graph_.isProtected[v] = true;
        if (deg > 2)
          ++stats_.numJunctions;
        else if (deg == 1)
          ++stats_.numEndpoints;
      }
    }
  }

  // Detect sharp corners at degree-2 vertices
  void detectSharpCorners() {
    NumericType cosThreshold = std::cos(angleThreshold_ * M_PI / 180.0);

    for (size_t v = 0; v < graph_.points.size(); ++v) {
      if (graph_.isDeleted[v] || graph_.isProtected[v])
        continue;
      if (graph_.degree[v] != 2)
        continue;

      // Get the two neighbors
      std::vector<unsigned> neighbors;
      for (unsigned n : graph_.adj[v]) {
        if (!graph_.isDeleted[n])
          neighbors.push_back(n);
      }
      if (neighbors.size() != 2)
        continue;

      const auto &p = graph_.points[v];
      const auto &p1 = graph_.points[neighbors[0]];
      const auto &p2 = graph_.points[neighbors[1]];

      // Compute vectors from v to neighbors
      NumericType v1x = p1[0] - p[0], v1y = p1[1] - p[1];
      NumericType v2x = p2[0] - p[0], v2y = p2[1] - p[1];

      NumericType len1 = std::sqrt(v1x * v1x + v1y * v1y);
      NumericType len2 = std::sqrt(v2x * v2x + v2y * v2y);

      if (len1 < 1e-12 || len2 < 1e-12)
        continue;

      // Normalize
      v1x /= len1;
      v1y /= len1;
      v2x /= len2;
      v2y /= len2;

      // Dot product gives cos(angle)
      NumericType cosAngle = v1x * v2x + v1y * v2y;

      // If angle is sharp (cos > threshold means angle < threshold from 180°)
      // We want to detect corners, so we check if deviation from 180° is large
      // cos(180°) = -1, so sharp corner means cos > -cosThreshold
      if (cosAngle > -cosThreshold) {
        graph_.isProtected[v] = true;
      }
    }
  }

  // Compute edge length statistics
  void computeEdgeLengthStats() {
    std::vector<NumericType> lengths;
    lengths.reserve(graph_.edges.size());

    for (const auto &e : graph_.edges) {
      if (graph_.isDeleted[e[0]] || graph_.isDeleted[e[1]])
        continue;
      lengths.push_back(distance(graph_.points[e[0]], graph_.points[e[1]]));
    }

    if (lengths.empty())
      return;

    std::sort(lengths.begin(), lengths.end());
    stats_.minEdgeLength = lengths.front();
    stats_.maxEdgeLength = lengths.back();
    stats_.medianEdgeLength = lengths[lengths.size() / 2];

    // Auto-compute parameters if not set
    if (hTarget_ < 0) {
      hTarget_ = stats_.medianEdgeLength;
    }
    if (epsMerge_ < 0) {
      epsMerge_ = 0.03 * hTarget_;
    }
    if (lMin_ < 0) {
      lMin_ = 0.25 * hTarget_;
    }
    if (simplifyTol_ < 0) {
      simplifyTol_ = 0.02 * hTarget_;
    }

    // Count edges below l_min
    for (NumericType len : lengths) {
      if (len < lMin_)
        ++stats_.edgesBelowLMin;
    }
  }

  // Stage 2: Extract branches (polyline decomposition)
  void extractBranches() {
    branches_.clear();
    std::set<std::pair<unsigned, unsigned>> visitedEdges;

    // Find all starting points: protected vertices or any unvisited edge
    for (size_t v = 0; v < graph_.points.size(); ++v) {
      if (graph_.isDeleted[v])
        continue;
      if (!graph_.isProtected[v])
        continue;

      // Start branches from this protected vertex
      for (unsigned neighbor : graph_.adj[v]) {
        if (graph_.isDeleted[neighbor])
          continue;

        auto edgeKey = std::make_pair(std::min((unsigned)v, neighbor),
                                      std::max((unsigned)v, neighbor));
        if (visitedEdges.count(edgeKey))
          continue;

        // Start a new branch
        Branch branch;
        branch.vertices.push_back(v);
        unsigned current = v;
        unsigned next = neighbor;

        while (true) {
          auto ek =
              std::make_pair(std::min(current, next), std::max(current, next));
          visitedEdges.insert(ek);
          branch.vertices.push_back(next);

          if (graph_.isProtected[next]) {
            // End of branch
            break;
          }

          // Find next vertex (the other neighbor of a degree-2 vertex)
          unsigned prev = current;
          current = next;
          next = std::numeric_limits<unsigned>::max();

          for (unsigned n : graph_.adj[current]) {
            if (graph_.isDeleted[n])
              continue;
            if (n != prev) {
              next = n;
              break;
            }
          }

          if (next == std::numeric_limits<unsigned>::max()) {
            break; // dead end (shouldn't happen for valid graph)
          }
        }

        if (branch.vertices.size() >= 2) {
          branches_.push_back(std::move(branch));
          ++stats_.numBranches;
        }
      }
    }

    // Handle pure cycles (all vertices have degree 2)
    for (size_t v = 0; v < graph_.points.size(); ++v) {
      if (graph_.isDeleted[v])
        continue;
      if (graph_.isProtected[v])
        continue;
      if (graph_.degree[v] != 2)
        continue;

      // Check if this vertex is part of an unvisited cycle
      bool allVisited = true;
      for (unsigned neighbor : graph_.adj[v]) {
        if (graph_.isDeleted[neighbor])
          continue;
        auto ek = std::make_pair(std::min((unsigned)v, neighbor),
                                 std::max((unsigned)v, neighbor));
        if (!visitedEdges.count(ek)) {
          allVisited = false;
          break;
        }
      }

      if (allVisited)
        continue;

      // Start a cycle from this vertex
      Branch cycle;
      cycle.isCycle = true;
      cycle.vertices.push_back(v);

      unsigned start = v;
      unsigned current = v;
      unsigned next = std::numeric_limits<unsigned>::max();

      // Pick first unvisited neighbor
      for (unsigned neighbor : graph_.adj[v]) {
        if (graph_.isDeleted[neighbor])
          continue;
        auto ek = std::make_pair(std::min((unsigned)v, neighbor),
                                 std::max((unsigned)v, neighbor));
        if (!visitedEdges.count(ek)) {
          next = neighbor;
          break;
        }
      }

      if (next == std::numeric_limits<unsigned>::max())
        continue;

      while (next != start) {
        auto ek =
            std::make_pair(std::min(current, next), std::max(current, next));
        visitedEdges.insert(ek);
        cycle.vertices.push_back(next);

        unsigned prev = current;
        current = next;
        next = std::numeric_limits<unsigned>::max();

        for (unsigned n : graph_.adj[current]) {
          if (graph_.isDeleted[n])
            continue;
          if (n != prev) {
            next = n;
            break;
          }
        }

        if (next == std::numeric_limits<unsigned>::max())
          break;
      }

      // Close the cycle
      if (next == start) {
        auto ek =
            std::make_pair(std::min(current, start), std::max(current, start));
        visitedEdges.insert(ek);
        cycle.vertices.push_back(start); // close the loop
      }

      if (cycle.vertices.size() >= 3) {
        branches_.push_back(std::move(cycle));
        ++stats_.numCycles;
      }
    }
  }

  // Count connected components
  void countComponents() {
    std::vector<bool> visited(graph_.points.size(), false);
    stats_.numComponents = 0;

    for (size_t v = 0; v < graph_.points.size(); ++v) {
      if (graph_.isDeleted[v] || visited[v])
        continue;
      if (graph_.degree[v] == 0)
        continue; // skip isolated

      // BFS
      std::queue<unsigned> q;
      q.push(v);
      visited[v] = true;

      while (!q.empty()) {
        unsigned curr = q.front();
        q.pop();

        for (unsigned neighbor : graph_.adj[curr]) {
          if (!graph_.isDeleted[neighbor] && !visited[neighbor]) {
            visited[neighbor] = true;
            q.push(neighbor);
          }
        }
      }
      ++stats_.numComponents;
    }
  }

  // Stage 4: Merge near-duplicate vertices
  void mergeNearDuplicates() {
    NumericType epsSq = epsMerge_ * epsMerge_;

    // Simple O(n²) approach for now - could use spatial hashing for large
    // inputs
    std::vector<unsigned> mergeTarget(graph_.points.size());
    std::iota(mergeTarget.begin(), mergeTarget.end(), 0);

    for (size_t i = 0; i < graph_.points.size(); ++i) {
      if (graph_.isDeleted[i])
        continue;

      for (size_t j = i + 1; j < graph_.points.size(); ++j) {
        if (graph_.isDeleted[j])
          continue;

        NumericType dSq = distanceSquared(graph_.points[i], graph_.points[j]);
        if (dSq > epsSq)
          continue;

        // Decide which to keep
        bool iProtected = graph_.isProtected[i];
        bool jProtected = graph_.isProtected[j];

        if (iProtected && jProtected) {
          // Both protected: only merge if extremely close
          if (dSq > 1e-24)
            continue;
        }

        // Merge j into i if i is protected, else merge into whichever
        unsigned keep = iProtected ? i : (jProtected ? j : i);
        unsigned remove = (keep == i) ? j : i;

        // Rewire edges from remove to keep
        for (unsigned neighbor : graph_.adj[remove]) {
          if (graph_.isDeleted[neighbor])
            continue;
          if (neighbor == keep)
            continue;

          // Add edge keep-neighbor if not exists
          bool exists = false;
          for (unsigned n : graph_.adj[keep]) {
            if (n == neighbor) {
              exists = true;
              break;
            }
          }
          if (!exists) {
            graph_.adj[keep].push_back(neighbor);
            graph_.adj[neighbor].push_back(keep);
            ++graph_.degree[keep];
            // Replace 'remove' with 'keep' in neighbor's adjacency
            for (auto &n : graph_.adj[neighbor]) {
              if (n == remove) {
                n = keep;
                break;
              }
            }
          }
        }

        // Mark as deleted
        graph_.isDeleted[remove] = true;
        graph_.degree[remove] = 0;
        graph_.adj[remove].clear();
        mergeTarget[remove] = keep;

        // Transfer protection status
        if (jProtected && keep == i) {
          graph_.isProtected[i] = true;
        }

        ++stats_.mergedVertices;
      }
    }

    // Update edges
    rebuildEdges();
  }

  // Rebuild edge list from adjacency
  void rebuildEdges() {
    std::set<std::pair<unsigned, unsigned>> uniqueEdges;
    graph_.edges.clear();

    for (size_t v = 0; v < graph_.adj.size(); ++v) {
      if (graph_.isDeleted[v])
        continue;

      for (unsigned neighbor : graph_.adj[v]) {
        if (graph_.isDeleted[neighbor])
          continue;

        auto canonical = std::make_pair(std::min((unsigned)v, neighbor),
                                        std::max((unsigned)v, neighbor));
        if (!uniqueEdges.count(canonical)) {
          uniqueEdges.insert(canonical);
          graph_.edges.push_back({(unsigned)v, neighbor});
        }
      }
    }
  }

  // Stage 5: Collapse short edges
  void collapseShortEdges() {
    NumericType lMinSq = lMin_ * lMin_;
    bool changed = true;
    int maxIter = 100;

    while (changed && maxIter-- > 0) {
      changed = false;

      for (size_t i = 0; i < graph_.edges.size(); ++i) {
        unsigned u = graph_.edges[i][0];
        unsigned v = graph_.edges[i][1];

        if (graph_.isDeleted[u] || graph_.isDeleted[v])
          continue;

        NumericType dSq = distanceSquared(graph_.points[u], graph_.points[v]);
        if (dSq >= lMinSq)
          continue;

        bool uProtected = graph_.isProtected[u];
        bool vProtected = graph_.isProtected[v];

        if (uProtected && vProtected) {
          // Keep edge between two protected vertices
          continue;
        }

        // Collapse: keep the protected one, or u if neither protected
        unsigned keep = uProtected ? u : v;
        unsigned remove = (keep == u) ? v : u;

        // Move to midpoint if neither protected
        if (!uProtected && !vProtected) {
          graph_.points[keep][0] =
              (graph_.points[u][0] + graph_.points[v][0]) / 2;
          graph_.points[keep][1] =
              (graph_.points[u][1] + graph_.points[v][1]) / 2;
        }

        // Rewire edges
        for (unsigned neighbor : graph_.adj[remove]) {
          if (graph_.isDeleted[neighbor] || neighbor == keep)
            continue;

          bool exists = false;
          for (unsigned n : graph_.adj[keep]) {
            if (n == neighbor) {
              exists = true;
              break;
            }
          }

          if (!exists) {
            graph_.adj[keep].push_back(neighbor);
            ++graph_.degree[keep];
          }

          // Update neighbor's adjacency
          for (auto &n : graph_.adj[neighbor]) {
            if (n == remove) {
              n = keep;
              break;
            }
          }
        }

        // Remove the edge between keep and remove from keep's adjacency
        graph_.adj[keep].erase(std::remove(graph_.adj[keep].begin(),
                                           graph_.adj[keep].end(), remove),
                               graph_.adj[keep].end());
        --graph_.degree[keep];

        graph_.isDeleted[remove] = true;
        graph_.degree[remove] = 0;
        graph_.adj[remove].clear();

        ++stats_.collapsedEdges;
        changed = true;
      }

      if (changed) {
        rebuildEdges();
        // Recompute degrees
        for (size_t v = 0; v < graph_.points.size(); ++v) {
          if (!graph_.isDeleted[v]) {
            graph_.degree[v] = 0;
            for (unsigned n : graph_.adj[v]) {
              if (!graph_.isDeleted[n])
                ++graph_.degree[v];
            }
          }
        }
      }
    }
  }

  // Stage 7: Resample branches to target spacing
  void resampleBranches() {
    for (auto &branch : branches_) {
      if (branch.vertices.size() < 2)
        continue;

      std::vector<unsigned> newVertices;
      newVertices.push_back(branch.vertices[0]);

      for (size_t i = 0; i < branch.vertices.size() - 1; ++i) {
        unsigned u = branch.vertices[i];
        unsigned v = branch.vertices[i + 1];

        if (graph_.isDeleted[u] || graph_.isDeleted[v])
          continue;

        const auto &p1 = graph_.points[u];
        const auto &p2 = graph_.points[v];
        NumericType segLen = distance(p1, p2);

        if (segLen <= hTarget_ * 1.2) {
          // Segment is short enough, keep as is
          if (newVertices.back() != v) {
            newVertices.push_back(v);
          }
        } else {
          // Need to insert points
          int numSegments = static_cast<int>(std::ceil(segLen / hTarget_));
          NumericType step = segLen / numSegments;

          NumericType dx = (p2[0] - p1[0]) / segLen;
          NumericType dy = (p2[1] - p1[1]) / segLen;

          for (int j = 1; j < numSegments; ++j) {
            NumericType t = j * step;
            Point2D newPt = {p1[0] + t * dx, p1[1] + t * dy};

            // Check if too close to a protected vertex
            bool tooClose = false;
            if (graph_.isProtected[u]) {
              if (distance(newPt, p1) < epsMerge_)
                tooClose = true;
            }
            if (graph_.isProtected[v]) {
              if (distance(newPt, p2) < epsMerge_)
                tooClose = true;
            }

            if (!tooClose) {
              // Add new vertex
              unsigned newIdx = graph_.points.size();
              graph_.points.push_back(newPt);
              graph_.adj.push_back({});
              graph_.degree.push_back(0);
              graph_.isProtected.push_back(false);
              graph_.isDeleted.push_back(false);

              newVertices.push_back(newIdx);
              ++stats_.insertedPoints;
            }
          }

          if (newVertices.back() != v) {
            newVertices.push_back(v);
          }
        }
      }

      branch.vertices = std::move(newVertices);
    }
  }

  // Build output from branches
  void buildOutput() {
    output_.points.clear();
    output_.edges.clear();
    output_.polylines.clear();

    // Map old vertex indices to new compact indices
    std::unordered_map<unsigned, unsigned> oldToNew;
    unsigned nextIdx = 0;

    for (const auto &branch : branches_) {
      std::vector<unsigned> polyline;

      for (unsigned v : branch.vertices) {
        if (graph_.isDeleted[v])
          continue;

        unsigned newIdx;
        auto it = oldToNew.find(v);
        if (it == oldToNew.end()) {
          newIdx = nextIdx++;
          oldToNew[v] = newIdx;
          output_.points.push_back({graph_.points[v][0], graph_.points[v][1]});
        } else {
          newIdx = it->second;
        }

        // Avoid duplicate consecutive vertices
        if (polyline.empty() || polyline.back() != newIdx) {
          polyline.push_back(newIdx);
        }
      }

      if (polyline.size() >= 2) {
        // Add edges for this polyline
        for (size_t i = 0; i < polyline.size() - 1; ++i) {
          output_.edges.push_back(canonicalize(polyline[i], polyline[i + 1]));
        }
        output_.polylines.push_back(std::move(polyline));
      }
    }

    // Remove duplicate edges
    std::set<std::pair<unsigned, unsigned>> uniqueEdges;
    std::vector<Edge> dedupedEdges;
    for (const auto &e : output_.edges) {
      auto key = std::make_pair(e[0], e[1]);
      if (!uniqueEdges.count(key)) {
        uniqueEdges.insert(key);
        dedupedEdges.push_back(e);
      }
    }
    output_.edges = std::move(dedupedEdges);

    stats_.outputPoints = output_.points.size();
    stats_.outputEdges = output_.edges.size();
  }

public:
  ConstraintCleaner() = default;

  /// Set input points (2D coordinates)
  void setPoints(const std::vector<Point2D> &points) { inputPoints_ = points; }

  /// Set input points from 3D mesh nodes (uses x, y)
  void setPoints(const std::vector<Vec3D<NumericType>> &nodes) {
    inputPoints_.clear();
    inputPoints_.reserve(nodes.size());
    for (const auto &n : nodes) {
      inputPoints_.push_back({n[0], n[1]});
    }
  }

  /// Set input edges
  void setEdges(const std::vector<Edge> &edges) { inputEdges_ = edges; }

  /// Set target edge spacing (if < 0, auto-computed from median edge length)
  void setTargetSpacing(NumericType h) { hTarget_ = h; }

  /// Set merge threshold for near-duplicate vertices
  void setMergeThreshold(NumericType eps) { epsMerge_ = eps; }

  /// Set minimum edge length threshold
  void setMinEdgeLength(NumericType lMin) { lMin_ = lMin; }

  /// Set simplification tolerance
  void setSimplificationTolerance(NumericType tol) { simplifyTol_ = tol; }

  /// Enable/disable polyline simplification (RDP-style)
  void setEnableSimplification(bool enable) { enableSimplification_ = enable; }

  /// Enable/disable resampling to uniform edge lengths
  void setEnableResampling(bool enable) { enableResampling_ = enable; }

  /// Set angle threshold for sharp corner detection (degrees)
  void setAngleThreshold(NumericType degrees) { angleThreshold_ = degrees; }

  /// Enable verbose output
  void setVerbose(bool verbose) { verbose_ = verbose; }

  /// Run the constraint cleaning pipeline
  void apply() {
    stats_ = ConstraintCleanerStats();
    stats_.inputPoints = inputPoints_.size();
    stats_.inputEdges = inputEdges_.size();

    // Stage 1: Normalize input
    normalizeInput();

    // Mark protected vertices
    markProtectedVertices();

    // Detect sharp corners
    detectSharpCorners();

    // Compute edge length stats and auto-tune parameters
    computeEdgeLengthStats();

    // Count components
    countComponents();

    // Stage 4: Merge near-duplicates
    mergeNearDuplicates();

    // Update protection after merge
    markProtectedVertices();

    // Stage 5: Collapse short edges
    collapseShortEdges();

    // Update protection after collapse
    markProtectedVertices();

    // Stage 2: Extract branches (after cleanup)
    extractBranches();

    // Stage 7: Resample (optional)
    if (enableResampling_) {
      resampleBranches();
    }

    // Build output
    buildOutput();

    if (verbose_) {
      stats_.print();
    }
  }

  /// Get the cleaned constraints
  const CleanedConstraints<NumericType> &getConstraints() const {
    return output_;
  }

  /// Get statistics
  const ConstraintCleanerStats &getStats() const { return stats_; }

  /// Apply cleaned constraints to a mesh (replaces nodes and lines)
  void applyToMesh(SmartPointer<viennals::Mesh<NumericType>> mesh) const {
    // Clear existing geometry
    auto &nodes = mesh->nodes;
    auto &lines = mesh->lines;
    nodes.clear();
    lines.clear();

    // Add points as 3D nodes (z=0)
    nodes.reserve(output_.points.size());
    for (const auto &p : output_.points) {
      nodes.push_back({p[0], p[1], NumericType(0)});
    }

    // Add edges as lines
    lines.reserve(output_.edges.size());
    for (const auto &e : output_.edges) {
      lines.push_back({e[0], e[1]});
    }
  }

  /// Get cleaned points as 3D nodes (z=0)
  std::vector<Vec3D<NumericType>> getNodesAs3D() const {
    std::vector<Vec3D<NumericType>> nodes;
    nodes.reserve(output_.points.size());
    for (const auto &p : output_.points) {
      nodes.push_back({p[0], p[1], NumericType(0)});
    }
    return nodes;
  }
};

} // namespace viennaps
