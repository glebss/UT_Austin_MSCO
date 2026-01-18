#include "tree.h"
#include <stdexcept>

int decide_on_quadrant(double x_center, double y_center, double xmin, double ymin, double xmax, double ymax) {
  double x_avg = (xmin + xmax) / 2.0;
  double y_avg = (ymin + ymax) / 2.0;
  if (x_center >= xmin && x_center < x_avg && y_center >= ymin && y_center < y_avg) {
    return 0;
  } else if (x_center >= x_avg && x_center <= xmax && y_center >= ymin && y_center < y_avg) {
    return 1;
  } else if (x_center >= xmin && x_center < x_avg && y_center >= y_avg && y_center <= ymax) {
    return 2;
  } else if (x_center >= x_avg && x_center <= xmax && y_center >= y_avg && y_center <= ymax ) {
    return 3;
  } else {
    throw std::runtime_error{"Unknown region"};
  }
  return -1;
}

void assign_bounds(Node* new_node, double xmin, double ymin, double xmax, double ymax, int quadrant) {
  double x_avg = (xmin + xmax) / 2.0;
  double y_avg = (ymax + ymin) / 2.0;
  switch (quadrant) {
    case 0:
        new_node->xmin = xmin;
        new_node->ymin = ymin;
        new_node->xmax = x_avg;
        new_node->ymax = y_avg;
        break;
    case 1:
        new_node->xmin = x_avg;
        new_node->ymin = ymin;
        new_node->xmax = xmax;
        new_node->ymax = y_avg;
        break;
    case 2:
        new_node->xmin = xmin;
        new_node->ymin = y_avg;
        new_node->xmax = x_avg;
        new_node->ymax = ymax;
        break;
    default:
        new_node->xmin = x_avg;
        new_node->ymin = y_avg;
        new_node->xmax = xmax;
        new_node->ymax = ymax;
        break;
  }
}

void construct_tree(Particle* objects, int num_objects, Tree* out_tree) {
  for (int i = 0; i < num_objects; ++i) {
    if (objects[i].mass < 0) { continue; }
    Node* cur_node = out_tree->root;
    while (true) {
      // first check if nullptr, i.e. the tree is empty
      if (!cur_node) {
        Node* new_node = new Node(objects[i].x, objects[i].y, objects[i].mass, true, objects[i].idx, 1);
        new_node->xmin = out_tree->xmin;
        new_node->ymin = out_tree->ymin;
        new_node->xmax = out_tree->xmax;
        new_node->ymax = out_tree->ymax;
        new_node->num_objects += 1;
        out_tree->root = new_node;
        out_tree->num_nodes += 1;
        break;
      }
      if (!cur_node->is_single) {
        cur_node->x_center = (cur_node->x_center * cur_node->mass + objects[i].mass * objects[i].x) / (cur_node->mass + objects[i].mass);
        cur_node->y_center = (cur_node->y_center * cur_node->mass + objects[i].y * objects[i].mass) / (cur_node->mass + objects[i].mass);
        cur_node->mass += objects[i].mass;
        cur_node->num_objects += 1;
        // decide on the next node
        int quadrant = decide_on_quadrant(objects[i].x, objects[i].y, cur_node->xmin, cur_node->ymin,
                                          cur_node->xmax, cur_node->ymax);
        if (cur_node->children[quadrant]) {
          cur_node = cur_node->children[quadrant];
          continue;
        } else {
          Node* new_node = new Node(objects[i].x, objects[i].y, objects[i].mass, true, objects[i].idx, 1);
          assign_bounds(new_node, cur_node->xmin, cur_node->ymin, cur_node->xmax, cur_node->ymax, quadrant);
          cur_node->children[quadrant] = new_node;
          out_tree->num_nodes += 1;
          break;
        }
      } else {
        int quadrant_existed = decide_on_quadrant(cur_node->x_center, cur_node->y_center, cur_node->xmin,
                                                  cur_node->ymin, cur_node->xmax, cur_node->ymax);
        int quadrant_new = decide_on_quadrant(objects[i].x, objects[i].y, cur_node->xmin, cur_node->ymin,
                                              cur_node->xmax, cur_node->ymax);

        if (quadrant_existed != quadrant_new) {
          // update current node and create the new one
          Node* new_node = new Node(cur_node->x_center, cur_node->y_center, cur_node->mass, true, cur_node->idx, 1);
          cur_node->idx = -1;
          cur_node->x_center = (cur_node->x_center * cur_node->mass + objects[i].x * objects[i].mass) / (cur_node->mass + objects[i].mass);
          cur_node->y_center = (cur_node->y_center * cur_node->mass + objects[i].y * objects[i].mass) / (cur_node->mass + objects[i].mass);
          cur_node->mass += objects[i].mass;
          cur_node->num_objects += 1;
          cur_node->is_single = false;
          assign_bounds(new_node, cur_node->xmin, cur_node->ymin, cur_node->xmax, cur_node->ymax, quadrant_existed);
          cur_node->children[quadrant_existed] = new_node;
          out_tree->num_nodes += 1;
          Node* new_node2 = new Node(objects[i].x, objects[i].y, objects[i].mass, true, objects[i].idx, 1);
          assign_bounds(new_node2, cur_node->xmin, cur_node->ymin, cur_node->xmax, cur_node->ymax, quadrant_new);
          cur_node->children[quadrant_new] = new_node2;
          out_tree->num_nodes += 1;
          break;
        } else {
          double x_existed = cur_node->x_center;
          double y_existed = cur_node->y_center;
          double mass_existed = cur_node->mass;
          int idx_existed = cur_node->idx;
          cur_node->x_center = (cur_node->x_center * cur_node->mass + objects[i].x * objects[i].mass) / (cur_node->mass + objects[i].mass);
          cur_node->y_center = (cur_node->y_center * cur_node->mass + objects[i].y * objects[i].mass) / (cur_node->mass + objects[i].mass);
          cur_node->mass += objects[i].mass;
          cur_node->num_objects += 1;
          cur_node->is_single = false;
          cur_node->idx = -1;
          while (quadrant_existed == quadrant_new) {
            Node* new_node = new Node(cur_node->x_center, cur_node->y_center, cur_node->mass, false, -1, cur_node->num_objects);
            assign_bounds(new_node, cur_node->xmin, cur_node->ymin, cur_node->xmax, cur_node->ymax, quadrant_new);
            cur_node->children[quadrant_existed] = new_node;
            out_tree->num_nodes += 1;
            cur_node = new_node;
            quadrant_existed = decide_on_quadrant(x_existed, y_existed, cur_node->xmin, cur_node->ymin, cur_node->xmax, cur_node->ymax);
            quadrant_new = decide_on_quadrant(objects[i].x, objects[i].y, cur_node->xmin, cur_node->ymin, cur_node->xmax, cur_node->ymax);
          }
          Node* new_node1 = new Node(x_existed, y_existed, mass_existed, true, idx_existed, 1);
          assign_bounds(new_node1, cur_node->xmin, cur_node->ymin, cur_node->xmax, cur_node->ymax, quadrant_existed);
          cur_node->children[quadrant_existed] = new_node1;
          out_tree->num_nodes += 1;

          Node* new_node2 = new Node(objects[i].x, objects[i].y, objects[i].mass, true, objects[i].idx, 1);
          assign_bounds(new_node2, cur_node->xmin, cur_node->ymin, cur_node->xmax, cur_node->ymax, quadrant_new);
          cur_node->children[quadrant_new] = new_node2;
          out_tree->num_nodes += 1;
          break;
        }
      }
    }
  }
}