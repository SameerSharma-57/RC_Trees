#ifndef RAKE_COMPRESS
#define RAKE_COMPRESS

#include "CSR_matrix.cu"
#include "rake.cu"
#include "compress.cu"



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

        // marking 2 degree nodes
        find_2_degree_kernel<<<(((vertices+1023)/1024)),1024>>>(c_gpu,vertices,d_gpu,g.cct,g.idx);
        cudaMemcpy(c, c_gpu, sizeof(Vertex)*vertices, cudaMemcpyDeviceToHost);
        cudaMemcpy(d_host, d_gpu, sizeof(Vertex)*vertices, cudaMemcpyDeviceToHost);
        // marking 2 degree nodes completed

        // copying the graph into cpu
        copy_csr_mat(g,host_graph,true);
        // copying complete

        // marking compressible vertices
        mark_compressed_vertices(host_graph,c,d_host,round);
        cudaMemcpy(c_gpu, c, sizeof(Vertex)*vertices, cudaMemcpyHostToDevice);
        cudaMemcpy(d_gpu, d_host, sizeof(Vertex)*vertices, cudaMemcpyHostToDevice);
        // marking done

        // printf("two degree found");

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
    }


    cudaMemcpy(d_host, d_gpu, sizeof(Vertex)*vertices, cudaMemcpyDeviceToHost);

    cudaFree(d_gpu);
    cudaFree(l_gpu);
    cudaFree(update_gpu);
   
    return d_host;

    

}
#endif