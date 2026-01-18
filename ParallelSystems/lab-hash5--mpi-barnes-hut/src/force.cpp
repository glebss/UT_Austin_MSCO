#include "force.h"
#include <cmath>
#include <stdexcept>

double calc_distance(double x1, double y1, double x2, double y2) {
    return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

void calculate_total_force_and_update_particle(Particle* particle, Tree* tree,
                                               double theta, double dt, double G, double rlimit) {
    double Fx = 0.0;
    double Fy = 0.0;
    if (!tree->root) {
        throw std::runtime_error{"Empty tree!"};
    }
    std::vector<Node*> stack;
    stack.reserve(tree->num_nodes);
    stack.push_back(tree->root);
    while (!stack.empty()) {
        Node* cur_node = stack.back();
        stack.pop_back();
        // check if this is the particle itself
        if (cur_node->is_single && cur_node->idx == particle->idx) {
            continue;
        }
        double dist = calc_distance(particle->x, particle->y, cur_node->x_center, cur_node->y_center);
        if (dist < rlimit) { dist = rlimit; }
        double dist3 = dist * dist * dist; 
        double width = cur_node->ymax - cur_node->ymin;
        if (dist / width > theta) {
            if (cur_node->is_single) {
                double dx = particle->x - cur_node->x_center;
                double dy = particle->y - cur_node->y_center;
                Fx += G * particle->mass * cur_node->mass * dx / dist3;
                Fy += G * particle->mass * cur_node->mass * dy / dist3;
            } else {
                for (auto child : cur_node->children) {
                    if (child) { stack.push_back(child);}
                }
            }
        } else {
            double dx = particle->x - cur_node->x_center;
            double dy = particle->y - cur_node->y_center;
            Fx += G * particle->mass * cur_node->mass * dx / dist3;
            Fy += G * particle->mass * cur_node->mass * dy / dist3;
        }
    }
    double ax = Fx / particle->mass;
    double ay = Fy / particle->mass;
    double dt2 = dt * dt;
    particle->x += particle->x_velocity * dt + 0.5 * ax * dt2;
    particle->y += particle->y_velocity * dt + 0.5 * ay * dt2;

    if (particle->x > 4.0 || particle->x < 0.0 || particle->y > 4.0 || particle->y < 0.0) {
        particle->mass = -1;
        return;
    }
    particle->x_velocity += ax * dt;
    particle->y_velocity += ay * dt;
}

void calculate_batch(Particle* objects, int start_idx, int end_idx, Tree* tree, double theta, double dt) {
    for (int i = start_idx; i < end_idx; ++i) {
        calculate_total_force_and_update_particle(&objects[i], tree, theta, dt);
    }
}