#include "io.h"
#include <fstream>

int read_data(const char* in_file, Particle** objects) {
  std::ifstream in;
	in.open(in_file);
	int num_particles;
	in >> num_particles;
  *objects = new Particle[num_particles];
  for (int i = 0; i < num_particles; ++i) {
    Particle p;
    in >> p.idx >> p.x >> p.y >> p.mass >> p.x_velocity >> p.y_velocity;
    (*objects)[i] = p;
  }
  return num_particles;
}

void write_data(const char* out_file, Particle* objects, int num_particles) {
  std::ofstream out;
  out.open(out_file, std::ofstream::trunc);
  out << num_particles << std::endl;
  for (int i = 0; i < num_particles; ++i) {
    out << objects[i].idx << " " << objects[i].x << " " << objects[i].y << " " << objects[i].mass << " " <<
           objects[i].x_velocity << " " << objects[i].y_velocity << std::endl;
  }
}