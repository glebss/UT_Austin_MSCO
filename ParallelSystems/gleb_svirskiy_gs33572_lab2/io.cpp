#include "io.hpp"

void read_data(struct options_t* args, Dataset& dataset) {
    std::ifstream in;
	in.open(args->in_file);
    int num_points;
	in >> num_points;

    dataset.set_data(num_points, args->num_features);

    int num_point;
    for (int i = 0; i < num_points; ++i) {
        in >> num_point;
        for (int j = 0; j < args->num_features; ++j) {
            in >> dataset.get_data()[i * args->num_features + j];
        }
    }
    in.close();
}