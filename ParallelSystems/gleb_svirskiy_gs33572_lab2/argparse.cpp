#include "argparse.hpp"

void get_opts(int argc,
              char **argv,
              struct options_t *opts)
{
    if (argc == 1)
    {
        std::cout << "Usage:" << std::endl;
        std::cout << "\t-k <num_cluster>" << std::endl;
        std::cout << "\t-d <dims>" << std::endl;
        std::cout << "\t-i <input filename>" << std::endl;
        std::cout << "\t-m <maximum number of iterations>" << std::endl;
        std::cout << "\t-t <threshold>" << std::endl;
        std::cout << "\t[Optional] -c <output centroids>" << std::endl;
        std::cout << "\t-s <seed>" << std::endl;
        std::cout << "\t[Optional]--cuda <use cuda implementation, can't be used together with thrust>" << std::endl;
        std::cout << "\t[Optional]--thrust <use thrust implementation, can't be used together with cuda>" << std::endl;
        exit(0);
    }

    opts->out_centroids = false;
    opts->use_cuda = false;
    opts->use_thrust = false;

    struct option l_opts[] = {
        {"k", required_argument, NULL, 'k'},
        {"d", required_argument, NULL, 'd'},
        {"i", required_argument, NULL, 'i'},
        {"m", required_argument, NULL, 'm'},
        {"t", required_argument, NULL, 't'},
        {"c", no_argument, NULL, 'c'},
        {"s", required_argument, NULL, 's'},
        {"cuda", no_argument, NULL, 'u'},
        {"thrust", no_argument, NULL, 'h'}
    };

    int ind, c;
    while ((c = getopt_long(argc, argv, "k:d:i:m:t:cs:uh", l_opts, &ind)) != -1)
    {
        switch (c)
        {
        case 0:
            break;
        case 'k':
            opts->num_centroids = atoi((char *)optarg);
            break;
        case 'd':
            opts->num_features = atoi((char *)optarg);
            break;
        case 'i':
            opts->in_file = (char *)optarg;
            break;
        case 'm':
            opts->max_num_iter = atoi((char *)optarg);
            break;
        case 't':
            opts->threshold = atof((char *)optarg);
            break;
        case 'c':
            opts->out_centroids = true;
            break;
        case 's':
            opts->seed = atoi((char *)optarg);
            break;
        case 'u':
            opts->use_cuda = true;
            break;
        case 'h':
            opts->use_thrust = true;
            break;
        case ':':
            std::cerr << argv[0] << ": option -" << (char)optopt << "requires an argument." << std::endl;
            exit(1);
        }
    }
}
