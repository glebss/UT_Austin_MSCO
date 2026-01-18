#include "argparse.h"
#include <getopt.h>
#include <iostream>

void get_opts(int argc, char** argv, options_t* opts) {
    if (argc == 1)
    {
        std::cout << "Usage:" << std::endl;
        std::cout << "\t-i <file_path>" << std::endl;
        std::cout << "\t-o <file_path>" << std::endl;
        std::cout << "\t-s <steps>" << std::endl;
        std::cout << "\t-t <threshold>" << std::endl;
        std::cout << "\t-d <timestep>" << std::endl;
        std::cout << "\t[Optional] -V" << std::endl;
        exit(0);
    }

    opts->visualize = false;

    struct option l_opts[] = {
        {"in", required_argument, NULL, 'i'},
        {"out", required_argument, NULL, 'o'},
        {"n_steps", required_argument, NULL, 's'},
        {"threshold", required_argument, NULL, 't'},
        {"timestep", required_argument, NULL, 'd'},
        {"visualize", no_argument, NULL, 'V'}
    };

    int ind, c;
    while ((c = getopt_long(argc, argv, "i:o:s:t:d:V", l_opts, &ind)) != -1)
    {
        switch (c)
        {
        case 0:
            break;
        case 'i':
            opts->in_file = (char *)optarg;
            break;
        case 'o':
            opts->out_file = (char *)optarg;
            break;
        case 's':
            opts->n_steps = atoi((char *)optarg);
            break;
        case 't':
            opts->theta = atof((char *)optarg);
            break;
        case 'd':
            opts->dt = atof((char *)optarg);
            break;
        case 'V':
            opts->visualize = true;
            break;
        case ':':
            std::cerr << argv[0] << ": option -" << (char)optopt << "requires an argument." << std::endl;
            exit(1);
        }
    }
}