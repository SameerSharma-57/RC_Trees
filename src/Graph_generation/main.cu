
#include "Square_Matrix.cu"

#include "generate.cu"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <argp.h>



struct arguments
{
    char *args[2];
    std::string input_file, output_file;
    bool gen;
    Vertex vertices, edges;
};

static struct argp_option options[] = {
    {"gen", 'g', "GENERATE", 0, "Want to generate a new graph?"},
    {"vertices", 'v', "N_VETRTICES", 0,
     "Number of vertices you want in generated graph"},
    {"edges", 'e', "N_EDGES", 0, "Number of edges you want in generated graph"},
    {"output", 'o', "OUTFILE", 0, "path of file to give output of the program"},
    {"input", 'i', "INPUT_FILE", 0, "path of file to take input of graph"},
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
    case ARGP_KEY_ARG:
        if (state->arg_num >= 8)
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
    arguments.input_file = "/home/sameer/practice/Graphs/graph_1.txt";
    arguments.output_file = "";

    argp_parse(&argp, argc, argv, 0, 0, &arguments);
    srand(clock());
    // bool OnCPU = true;
    if (arguments.output_file !=
        "" ){ freopen(arguments.output_file.c_str(), "w", stdout); }
    string temp;

    cout<<"gen "<<arguments.gen<<endl;
    cout<<"vertices "<<arguments.vertices<<endl;
    cout<<"edges "<<arguments.edges<<endl;
    cout<<"input_file "<<arguments.input_file<<endl;
    cout<<"output_file "<<arguments.output_file<<endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    
    // cout<<"Stage 1 reached";
    bool success;
    Graph g =
        ReadGraphFromFile(arguments.input_file,success, false);

    if(!success){
        string error="Graph Allocation could not be done\n";
        cout<<error;
        return -1;
    }
    
    
    // cout<<"DONE"<<endl;
    return 0;
}