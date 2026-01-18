#pragma once
#include <vector>

struct Particle {
  double x;
  double y;
  double mass;
  int idx;
  double x_velocity;
  double y_velocity;
};


struct Node {
  Node() = default;
  Node(double x_center_, double y_center_, double mass_,
       bool is_single_ = false, int idx_ = -1, int num_objects_ = 1) {
      x_center = x_center_;
      y_center = y_center_;
      mass = mass_;
      is_single = is_single_;
      idx = idx_;
  }
  ~Node() {
    for (auto child : children) {
        delete child;
    }
  }
  double xmin;
  double xmax;
  double ymin;
  double ymax;
  double x_center;
  double y_center;
  double mass;
  int num_objects{0};
  bool is_single;
  int idx{-1};
  std::vector<Node*> children{4, nullptr};
};

struct Tree {
  Node* root{nullptr};
  double xmin{0.0};
  double ymin{0.0};
  double xmax{4.0};
  double ymax{4.0};
  int num_nodes{0};
  ~Tree() {
    delete root;
  }
};

struct Batches {
  Batches() = default;
  Batches(int num_batches) {
    num_batches_ = num_batches;
    batches_starts.resize(num_batches_);
    batches_sizes.resize(num_batches_);
  }
  int num_batches_{0};
  std::vector<Particle*> batches_starts;
  std::vector<int> batches_sizes;
};

void construct_tree(Particle* objects, int num_objects, Tree* out_tree);