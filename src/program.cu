#include "CSR_matrix.cu"
#include "Cpu_timer.cpp"
#include "Cuda_timer.cu"
#include "generate.cu"
#include "rake.cu"
#include "rake_cpu.cpp"
#include "rake_compress2.cu"
#include "compress2.cu"
#include <argp.h>
#include <cstdio>
#include <iostream>

struct arguments
{
    char *args[1];
    std::string input_file, output_file;
    bool gen;
    Vertex vertices, edges;
    Vertex mode;
};

static struct argp_option options[] = {
    {"gen", 'g', "GENERATE", 0, "Want to generate a new graph?"},
    {"vertices", 'v', "N_VETRTICES", 0,
     "Number of vertices you want in generated graph"},
    {"edges", 'e', "N_EDGES", 0, "Number of edges you want in generated graph"},
    {"output", 'o', "OUTFILE", 0, "path of file to give output of the program"},
    {"input", 'i', "INPUT_FILE", 0, "path of file to take input of graph"},
    {"mode", 'm', "MODE", 0, "mode of operation for the program"},
    {0}

};

static error_t parse_opt(int key, char *arg, struct argp_state *state)
{
    struct arguments *argument = (arguments *)state->input;

    switch (key)
    {
    case 'g':
        argument->gen = 1;
        break;
    case 'v':
        argument->vertices = atoi(arg);
        break;
    case 'e':
        argument->edges = atoi(arg);
        break;
    case 'o':
        argument->output_file = arg;
        break;

    case 'i':
        argument->input_file = arg;
        break;
    case 'm':
        argument->mode = atoi(arg);
        break;
    case ARGP_KEY_ARG:
        if (state->arg_num >= 11)
        {
            argp_usage(state);
        }
        argument->args[state->arg_num] = arg;
        break;
    default:

        return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

static struct argp argp = {options, parse_opt, 0, 0};

int main(int argc, char **argv)
{

    struct arguments arguments;

    arguments.gen = 0;
    arguments.vertices = 5;
    arguments.edges = 10;
    arguments.input_file = "";
    arguments.output_file = "";
    arguments.mode = 0;
    argp_parse(&argp, argc, argv, 0, 0, &arguments);

    if (arguments.input_file == "" || arguments.output_file == "")
    {
        cout << "Please provide path for input and output files" << endl;
        return -1;
    }

    freopen(arguments.output_file.c_str(), "w", stdout);

    if (arguments.gen)
    {
        cout << "Generating tree for " << arguments.vertices << " vertices and "
             << (arguments.vertices - 1) << " edges" << endl;
        SaveRandomTreeToFile(arguments.vertices, arguments.input_file);
        cout << "Tree generation completed" << endl;
    }

    if(arguments.mode!=0)
    {

        CPU_timer cpu_timer;
        CUDA_timer cuda_timer;

        // string output_file = "/home/sameer/RC_trees/Graphs/graphs_3.txt";
        // SaveRandomTreeToFile(1e7,output_file);

        CSR_mat g_sparse = ReadSparseGraph(arguments.input_file, false);

        Vertex vertices = g_sparse.Get_Vertex_count();
        Vertex edges = g_sparse.Get_edge_count();

        CSR_mat g_parallel_sparse;
        g_parallel_sparse.Allocate(vertices, edges, false);
        cudaMemcpy(g_parallel_sparse.nnz, g_sparse.nnz,
                   sizeof(Weight) * edges * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(g_parallel_sparse.idx, g_sparse.idx,
                   sizeof(Vertex) * edges * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(g_parallel_sparse.cct, g_sparse.cct,
                   sizeof(Vertex) * (vertices + 1), cudaMemcpyHostToDevice);

        if (arguments.mode == 1)
        {
            Vertex *d_gpu;
            Vertex *d_cpu;


            cuda_timer.start();
            d_gpu = GenerateCompressedGraph(g_parallel_sparse);
            cudaDeviceSynchronize();
            cuda_timer.stop();

            cout << "time taken by parallel algorithm "
                 << cuda_timer.time_elapsed() << endl;
        }

        else if(arguments.mode==2){
            // compress_2(g_parallel_sparse,0);
        }
        // Print_array(d_cpu,vertices);
        // Print_array(d_gpu,vertices);

        g_parallel_sparse.Deallocate();
    }
    return 0;
}
