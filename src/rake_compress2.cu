#ifndef RAKE_COMPRESS
#define RAKE_COMPRESS

#include "CSR_matrix.cu"
#include "rake.cu"
#include "compress2.cu"
#include "bfs.cu"
#include <algorithm>



Vertex *GenerateCompressedGraph(const CSR_mat g){

    const Vertex vertices = g.Get_Vertex_count();
    const Vertex edges = g.Get_edge_count();
    Vertex round = 1;
    Vertex *d_host = (Vertex*)malloc(sizeof(Vertex)*vertices);
    

    int *l=(int *)malloc(sizeof(int)*vertices);
    memset(l,0,sizeof(int)*vertices);

    int *c=(int *)malloc(sizeof(int)*vertices);
    memset(c,0,sizeof(int)*vertices);

    Vertex *d_gpu;
    cudaMalloc(&d_gpu,sizeof(Vertex)*vertices);
    cudaMemset(d_gpu, 0, sizeof(Vertex)*vertices);

    int *l_gpu;
    cudaMalloc(&l_gpu,sizeof(int)*vertices);
    cudaMemset(l_gpu,0,sizeof(int)*vertices);

    int *c_gpu;
    cudaMalloc(&c_gpu,sizeof(int)*vertices);
    cudaMemset(c_gpu,0,sizeof(int)*vertices);

    Weight *dist;
    cudaMalloc(&dist,sizeof(Weight)*vertices);
    cudaMemset(dist,0,sizeof(Weight)*vertices);

    bool *update_gpu;
    cudaMalloc(&update_gpu,sizeof(bool));
    cudaMemset(update_gpu, 0, sizeof(bool));

    bool *update_host = (bool*)malloc(sizeof(bool));
    *update_host=true;

    CSR_mat intermediate_graph;
    intermediate_graph.Allocate(g.vertex_count, g.edge_count,false);

    CSR_mat host_graph;
    host_graph.Allocate(g.vertex_count,g.edge_count);


    

    while(*update_host){
        cudaMemset(update_gpu, false, sizeof(bool));

        //rake process
        find_leaf_kernel<<<(((vertices+1023)/1024)),1024>>>(l_gpu,vertices,d_gpu,g.cct,g.idx);
        cudaMemcpy(l, l_gpu, sizeof(Vertex)*vertices, cudaMemcpyDeviceToHost);
        Compute<<<((vertices+1023)/1024),1024>>>(round,g.cct,g.idx,l_gpu,d_gpu,vertices,update_gpu);
        // rake complete

        // printf("rake complete");

        //compress operation

        // getting the level
        auto bfs_result = Get_Bfs_parallel(g,0);
        int max_level = *max_element(bfs_result.dist,bfs_result.dist+vertices);
        cudaMemcpy(dist, bfs_result.dist, sizeof(Weight)*vertices, cudaMemcpyHostToDevice);
        // calculating level complete

        // marking compressible vertices
        for (int level = 1; level < max_level+1; level++)
        {
            find_2_degree_kernel<<<(((vertices+1023)/1024)),1024>>>(c_gpu,vertices,d_gpu,g.cct,g.idx,dist,level,round);
            cudaDeviceSynchronize();
        }
        // marking done

        // storing the intermediate state
        copy_csr_mat(g,intermediate_graph);
        // storing complete

        // printf("vertices marked");
        compress<<<(((vertices+1023)/1024)),1024>>>(intermediate_graph.cct,intermediate_graph.idx,intermediate_graph.nnz,
        g.cct,g.idx,g.nnz,
        d_gpu,vertices,c_gpu,update_gpu);

        // compress operation complete
        round++;
        cudaMemcpy(update_host, update_gpu, sizeof(bool), cudaMemcpyDeviceToHost);

        
        //compress operation
    }


    cudaMemcpy(d_host, d_gpu, sizeof(Vertex)*vertices, cudaMemcpyDeviceToHost);

    cudaFree(d_gpu);
    cudaFree(l_gpu);
    cudaFree(update_gpu);
   
    return d_host;

    

}
#endif