#include "CSR_matrix.cu"
#include "Cpu_timer.cpp"
#include "Cuda_timer.cu"
#include "generate.cu"
#include "rake.cu"
#include "rake_cpu.cpp"
#include "temp.cu"
#include <argp.h>
#include <cstdio>
#include <iostream>

int main()
{

    CUDA_timer cuda_timer;

    string input_file = "/home/sameer/RC_trees/Graphs/graph_2.txt";
    CSR_mat g_sparse_old = ReadSparseGraph(input_file, false);

    string output_file = "/home/sameer/RC_trees/output/timestamp.txt";
    freopen(output_file.c_str(), "w", stdout);

    Vertex vertices = g_sparse_old.Get_Vertex_count();
    Vertex edges = g_sparse_old.Get_edge_count();

    cout<<"No of vertices "<<vertices<<" No of edges "<<edges<<endl;
    cout<<"Time is given in milliseconds\n";

    CSR_mat g_parallel_sparse;
    g_parallel_sparse.Allocate(vertices, edges, false);
    cudaMemcpy(g_parallel_sparse.nnz, g_sparse_old.nnz,
               sizeof(Weight) * edges * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(g_parallel_sparse.idx, g_sparse_old.idx,
               sizeof(Vertex) * edges * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(g_parallel_sparse.cct, g_sparse_old.cct,
               sizeof(Vertex) * (vertices + 1), cudaMemcpyHostToDevice);

    CSR_mat g_sparse_new = ReadSparseGraph(input_file, false);
    CSR_mat g_parallel_sparse_new;
    g_parallel_sparse_new.Allocate(vertices, edges, false);
    cudaMemcpy(g_parallel_sparse_new.nnz, g_parallel_sparse.nnz,
               sizeof(Weight) * edges * 2, cudaMemcpyDeviceToDevice);
    cudaMemcpy(g_parallel_sparse_new.idx, g_parallel_sparse.idx,
               sizeof(Vertex) * edges * 2, cudaMemcpyDeviceToDevice);
    cudaMemcpy(g_parallel_sparse_new.cct, g_parallel_sparse.cct,
               sizeof(Vertex) * (vertices + 1), cudaMemcpyDeviceToDevice);

    Vertex round = 1;
    Vertex *d_host = (Vertex *)malloc(sizeof(Vertex) * vertices);

    int *l = (int *)malloc(sizeof(int) * vertices);
    memset(l, 0, sizeof(int) * vertices);

    int *c = (int *)malloc(sizeof(int) * vertices);
    memset(c, 0, sizeof(int) * vertices);

    Vertex *d_gpu;
    cudaMalloc(&d_gpu, sizeof(Vertex) * vertices);
    cudaMemset(d_gpu, 0, sizeof(Vertex) * vertices);

    int *l_gpu;
    cudaMalloc(&l_gpu, sizeof(int) * vertices);
    cudaMemset(l_gpu, 0, sizeof(int) * vertices);

    int *c_gpu;
    cudaMalloc(&c_gpu, sizeof(int) * vertices);
    cudaMemset(c_gpu, 0, sizeof(int) * vertices);

    bool *update_gpu;
    cudaMalloc(&update_gpu, sizeof(bool));
    cudaMemset(update_gpu, 0, sizeof(bool));

    bool *update_host = (bool *)malloc(sizeof(bool));
    *update_host = true;

    cudaMemset(update_gpu, false, sizeof(bool));

    cuda_timer.start();
    find_leaf_kernel<<<(((vertices + 1023) / 1024)), 1024>>>(
        l_gpu, vertices, d_gpu, g_parallel_sparse.cct, g_parallel_sparse.idx);
    cudaDeviceSynchronize();
    cuda_timer.stop();
    cout << "Time required to mark leaf vertices "
         << cuda_timer.time_elapsed_milliseconds() << endl;

    cudaMemcpy(l, l_gpu, sizeof(Vertex) * vertices, cudaMemcpyDeviceToHost);

    cuda_timer.start();
    Compute<<<((vertices + 1023) / 1024), 1024>>>(round, g_parallel_sparse.cct,
                                                  g_parallel_sparse.idx, l_gpu,
                                                  d_gpu, vertices, update_gpu);
    cudaDeviceSynchronize();
    cuda_timer.stop();
    cout << "Time requried to rake vertices "
         << cuda_timer.time_elapsed_milliseconds() << endl;
    cudaMemcpy(update_host, update_gpu, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_host, d_gpu, sizeof(Vertex) * vertices,
               cudaMemcpyDeviceToHost);

    cuda_timer.start();
    mark_compressed_vertices(g_sparse_old, c, d_host, round, update_host);
    cuda_timer.stop();
    cout << "Time requried to mark compressable vertices "
         << cuda_timer.time_elapsed_milliseconds() << endl;

    cudaMemcpy(c_gpu, c, sizeof(int) * vertices, cudaMemcpyHostToDevice);

    cuda_timer.start();
    compress<<<((vertices + 1023) / 1024), 1024>>>(
        g_parallel_sparse.cct, g_parallel_sparse.idx, g_parallel_sparse.nnz,
        g_parallel_sparse_new.cct, g_parallel_sparse_new.idx,
        g_parallel_sparse_new.nnz, d_gpu, vertices, c_gpu);
    cudaDeviceSynchronize();
    cuda_timer.stop();
    cout << "Time required to compress vertices "
         << cuda_timer.time_elapsed_milliseconds();

    cudaMemcpy(g_sparse_new.nnz, g_parallel_sparse_new.nnz,
               sizeof(Weight) * edges * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(g_sparse_new.idx, g_parallel_sparse_new.idx,
               sizeof(Vertex) * edges * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(g_sparse_new.cct, g_parallel_sparse_new.cct,
               sizeof(Vertex) * (vertices + 1), cudaMemcpyDeviceToHost);

    //     g_sparse_new.print_mat();
    

    return 0;
}