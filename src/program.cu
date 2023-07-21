#include <cstdio>   

#include "rake.cu"
#include "rake_cpu.cpp"
#include "generate.cu"





int main(){


    string input_file = "/home/sameer/RC_trees/Graphs/graph_3.txt";
    freopen("/home/sameer/RC_trees/Graphs/graph_output.txt","w",stdout);

    // string output_file = "/home/sameer/RC_trees/Graphs/graphs_3.txt";
    // SaveRandomTreeToFile(1e7,output_file);
   
   CSR_mat g_sparse =
        ReadSparseGraph(input_file, false);


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


    

    Vertex *d_gpu;
    Vertex *d_cpu;
    d_cpu = generateCompressedGraph(g_sparse);
    d_gpu = GenerateCompressedGraph(g_parallel_sparse);



    cout<<"are results same: "<<compareArr(d_cpu,d_gpu,vertices)<<endl;
    // Print_array(d_cpu,vertices);
    // Print_array(d_gpu,vertices);

    

    return 0;

}