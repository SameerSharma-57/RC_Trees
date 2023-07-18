#ifndef RAKE
#define RAKE

#include "CSR_matrix.cu"





__global__ void find_leaf_kernel(bool* l,const Vertex vertices,const Vertex* d, const Vertex* cct, const Vertex* idx){
    int tid = blockDim.x*blockIdx.x + threadIdx.x;

    if(tid<vertices && d[tid]==0){

        int n_neighbours=0;

        for (int i = cct[tid]; i < cct[tid+1]; i++)
        {
            if(d[idx[i]]==0){
                n_neighbours++;
            }
            if(n_neighbours>1){
                break;
            }
        }

        if(n_neighbours==1){
            l[tid]=1;
        }
        else{
            l[tid]=0;
        }
        
        
        
    }
}


__global__ void Compute(const Vertex round,const Vertex* cct,const Vertex* idx,const bool* l,Vertex* d, const Vertex vertices, bool* update){
    int tid = blockDim.x*blockIdx.x + threadIdx.x;

    if(tid<vertices && d[tid]==0){

        int n_neighbours=0;

        for (int i = cct[tid]; i < cct[tid+1]; i++)
        {
            if(d[idx[i]]==0){

                if(l[idx[i]]){
                    d[idx[i]]=round;
                    *update=true;
                }

                n_neighbours++;
            }
        }

        if(n_neighbours==0){
            d[tid]=round;
            *update=true;
        }
        
    }

}




Vertex * GenerateCompressedGraph(const CSR_mat g){

    Vertex vertices = g.Get_Vertex_count();
    Vertex edges = g.Get_edge_count();
    Vertex round = 1;
    Vertex* d_host = (Vertex*)malloc(sizeof(Vertex)*vertices);
    

    bool* l=(bool* )malloc(sizeof(bool)*vertices);

    Vertex* d_gpu;
    cudaMalloc(&d_gpu,sizeof(Vertex)*vertices);
    cudaMemset(&d_gpu, 0, sizeof(Vertex)*vertices);

    bool* l_gpu;
    cudaMalloc(&l_gpu,sizeof(bool)*vertices);

    bool* update_gpu;
    cudaMalloc(&update_gpu,sizeof(bool));
    cudaMemset(&update_gpu, 0, sizeof(bool));

    bool* update_host = (bool*)malloc(sizeof(bool));
    *update_host=true;

    

    while(update_host){
        find_leaf_kernel<<<(((vertices+1023)/1024)),1024>>>(l_gpu,vertices,d_gpu,g.cct,g.idx);
        Compute<<<((vertices+1023)/1024),1024>>>(round,g.cct,g.idx,l_gpu,d_gpu,vertices,update_gpu);
        round++;
        cudaMemcpy(&update_host, &update_gpu, sizeof(bool), cudaMemcpyDeviceToHost);
    }


    cudaMemcpy(&d_host, &d_gpu, sizeof(Vertex)*vertices, cudaMemcpyDeviceToHost);
    return d_host;

}
#endif