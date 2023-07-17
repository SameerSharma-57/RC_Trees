#include "CSR_matrix.cu"
#include "Square_Matrix.cu"
#include "bfs.cu"
#include "generate.cu"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <argp.h>

typedef BFS_result (*my_function) (const CSR_mat &g, const Vertex source, const Vertex dest);

bool Verify(BFS_result cpu, BFS_result gpu, vector<Vertex> &incorrect)
{
    Vertex vertex_count = cpu.vertex_count;
    bool out = true;
    for (int i = 0; i < vertex_count; i++)
    {
        if (cpu.dist[i] != gpu.dist[i])
        {
            cout << i << " " << cpu.dist[i] << " " << gpu.dist[i] << endl;
            incorrect.push_back(i);
            out = false;
        }
    }
    return out;
}

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

    // cout<<"gen "<<arguments.gen<<endl;
    // cout<<"vertices "<<arguments.vertices<<endl;
    // cout<<"edges "<<arguments.edges<<endl;
    // cout<<"input_file "<<arguments.input_file<<endl;
    // cout<<"output_file "<<arguments.output_file<<endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if (arguments.gen)
    {
        cout << "generating graph\n";
        SaveRandomGraphToFile(arguments.vertices, arguments.edges,
                              arguments.input_file);
    }
    // cout<<"Stage 1 reached";
    bool success;
    Graph g =
        ReadGraphFromFile(arguments.input_file,success, false);

    if(!success){
        string error="Graph Allocation could not be done\n";
        cout<<error;
        return -1;
    }
    // arguments.vertices = g.Get_Vertex_count();
    // arguments.edges=g.Get_Edge_count();
    // // g.adjmat.print_matrix();

    // // g.Allocate(arguments.vertices, OnCPU);

    // // g.Set_Edge(0, 4, 2);
    // // g.Set_Edge(1, 3, 4);
    // // g.Set_Edge(2, 1, 6);
    // // g.Set_Edge(0, 1, 2);

    // Graph g_parallel;
    // g_parallel.Allocate(arguments.vertices, false);
    // cudaMemcpy(g_parallel.adjmat.Array, g.adjmat.Array,
    //            sizeof(Weight) * arguments.vertices * arguments.vertices,
    //            cudaMemcpyHostToDevice);
    BFS_result result = Get_Bfs(g, 0, arguments.vertices);
    // BFS_result result_parallel = Get_Bfs_parallel(g_parallel, 0, arguments.vertices);

    CSR_mat g_sparse =
        ReadSparseGraph(arguments.input_file, false);
    // // g_sparse.print_mat();
        arguments.vertices = g_sparse.Get_Vertex_count();
    arguments.edges=g_sparse.Get_edge_count();
    // cout<<"vertices "<<arguments.vertices<<" edges "<<arguments.edges<<endl;

    BFS_result result_sparse = Get_Bfs(g_sparse, 0, arguments.vertices);

    CSR_mat g_parallel_sparse;
    g_parallel_sparse.Allocate(arguments.vertices, arguments.edges, false);
    cudaMemcpy(g_parallel_sparse.nnz, g_sparse.nnz,
               sizeof(Weight) * arguments.edges * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(g_parallel_sparse.idx, g_sparse.idx,
               sizeof(Vertex) * arguments.edges * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(g_parallel_sparse.cct, g_sparse.cct,
               sizeof(Vertex) * (arguments.vertices + 1), cudaMemcpyHostToDevice);


    cudaEventRecord(start);
    BFS_result result_parallel_sparse =
        Get_Bfs_parallel(g_parallel_sparse, 0, arguments.vertices);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);

    float time;
    cudaEventElapsedTime(&time, start, stop);
    cout<<"Time taken by parallel implentation of csr matrix is: "<<time<<endl;

    // cout << "\nCPU Implementation\nDist : ";
    // Print_array(result.dist, arguments.vertices);
    // cout << "\nPred : ";
    // Print_array(result.pred, arguments.vertices);

    // cout << "\nParallel Implementation\nDist : ";
    // Print_array(result_parallel.dist, arguments.vertices);
    // cout << "\nPred : ";
    // Print_array(result_parallel.pred, arguments.vertices);

    // result.Print_result();
    // result_sparse.Print_result();
    vector<Vertex> values;
    // if (Verify(result, result_parallel, values))
    // {
    //     cout << "\nresults from parallel and sequential implementation through "
    //             "adjacency matrix are identical\n";
    // }
    // else
    // {
    //     cout << "\nresults from parallel and sequential implementation through "
    //             "adjacency matrix are different\n";
    // }

    if (Verify(result, result_sparse, values))
    {
        cout << "\nresults from sequential implementation of adjacency and "
                "sparse are identical\n";
    }
    else
    {
        cout << "\nresults from sequential implementation of adjacency and "
                "sparse are different\n";
    }

    if (Verify(result_parallel_sparse, result_sparse, values))
    {
        cout << "\nresults from sequential and parallel implementation of "
                "sparse are identical\n";
    }
    else
    {
        cout << "\nresults from sequential and parallel implementation of "
                "sparse are different\n";
    }
    //     // for (auto x : values)
    //     // {
    //     //     printf("\nNow printing neighbours of %d ...\n", x);
    //     //     for (int i = 0; i < arguments.vertices; i++)
    //     //     {
    //     //         if (g.adjmat.Get(i, x))
    //     //             printf("%d ", i);
    //     //     }
    //     //     printf("\nNow printing neighbours of %d in CSR ...\n", x);
    //     //     for (int i = g_sparse.cct[x]; i < g_sparse.cct[x + 1]; i++)
    //     //     {
    //     //         printf("%d ", g_sparse.idx[i]);
    //     //     }
    //     // }
    // }

    // if (Verify(result, result_parallel_sparse, values))
    // {
    //     cout << "\nresults from sequential implementation through adjacency "
    //             "matrix and parallel implementation through CSR mat are "
    //             "identical\n";
    // }
    // else
    // {
    //     cout << "\nresults from sequential implementation through adjacency "
    //             "matrix and parallel implementation through CSR mat are "
    //             "different\n";
    // }

    // result.Deallocate();
    // result_parallel.Deallocate();
    // g.Deallocate();
    // g_parallel.Deallocate();
    // result_sparse.Print_result();
    // cout<<"\n\n\n";
    // result_parallel_sparse.Print_result();
    g_sparse.Deallocate();
    result_parallel_sparse.Deallocate();
    g_parallel_sparse.Deallocate();
    result_sparse.Deallocate();
    // cout<<"DONE"<<endl;
    return 0;
}