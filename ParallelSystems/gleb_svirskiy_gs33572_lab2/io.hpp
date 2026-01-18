#pragma once

#include "argparse.hpp"
#include "kmeans_cpu.hpp"
#include <iostream>
#include <fstream>

void read_data(struct options_t* args, Dataset& dataset);