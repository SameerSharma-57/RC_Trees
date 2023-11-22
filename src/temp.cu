
#include <cstdio>
#include "CSR_matrix.cu"

__global__ void myfunc(bool *l){
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if(tid<5){
        l[tid]=1;
    }
}

// int main(){


//     bool *l = (bool* )malloc(sizeof(bool)*5);

//     memset(l,0,sizeof(bool)*5);

//     bool *l_gpu;
//     cudaMalloc(&l_gpu,sizeof(bool)*5);
//     cudaMemset(l_gpu, 0, sizeof(bool)*5);
//     myfunc<<<10,1>>>(l_gpu);
//     cudaMemcpy(l, l_gpu, sizeof(bool)*5, cudaMemcpyDeviceToHost);
//     for (int i = 0; i < 5; i++)
//     {
//         printf("%d ",l[i]);
//     }
    

//     return 0;
// }

void mark_compressed_vertices(const CSR_mat old_mat, int* c, Vertex* d, const Vertex round, bool *update ){
    const int vertex = old_mat.Get_Vertex_count();
    
    int n_neighbours = 0;
    for (int v = 0; v < vertex; v++)
    {
        n_neighbours=0;   
        for (int i = old_mat.cct[v]; i < old_mat.cct[v+1]; i++)
        {
            
            if(d[old_mat.idx[i]]==0){
                n_neighbours++;
                if(n_neighbours>2){
                    break;
                }
            }
            
        }
	// __syncthreads();
        if (n_neighbours==2)
        {
            d[v]=round;
            c[v]=1;
            *update = true;
        }
        
        
    }
    

}

__global__ void find_2_degree_kernel(int *c,const Vertex vertices,const Vertex*d, const Vertex*cct, const Vertex*idx){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if(tid<vertices && d[tid]==0){

        int n_neighbours=0;

        for (int i = cct[tid]; i < cct[tid+1]; i++)
        {
            if(d[idx[i]]==0){
                n_neighbours+=1;
            }
            if(n_neighbours>2){
                break;
            }
        }

        c[tid] = (n_neighbours==2);
        
        
        
    }
}

__global__ void mark_compressible_nodes(int* c, const Vertex vertices, const Vertex * d, const Vertex*cct, const Vertex*idx){
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if(tid<vertices && c[tid]){
        int min = tid;
        for (int i = cct[tid]; i < cct[tid+1]; i++)
        {
           if(c[idx[i]]){
            min = tid<i? tid:i;
           }
        }

        if(min<tid){
            c[tid]=0;
        }
    }
}


// __global__ void compress(const CSR_mat &old, CSR_mat &new_mat, const Vertex* d, const Vertex vertices, const bool* c ){
__global__ void compress(const Vertex* old_cct, const Vertex* old_idx, const Weight* old_nnz, Vertex* new_cct, Vertex* new_idx, Weight* new_nnz, const Vertex* d, const Vertex vertices, const int*c ){
    
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if(tid<vertices && c[tid]){
        
        int ng[2];
        int ind=0;
        for (int i = old_cct[tid]; i < old_cct[tid+1]; i++)
        {
            if(d[old_idx[i]]==0){
                ng[ind++] = old_idx[i];
            }
        }

        int ind1=0;
        int ind2=0;

        for (int i = old_cct[ng[0]]; i < old_cct[ng[0]+1]; i++)
        {
            if(old_idx[i]==tid){
                ind1=i;
                break;
            }
        }

        for (int i = old_cct[ng[1]]; i < old_cct[ng[1]+1]; i++)
        {
            if(old_idx[i]==tid){
                ind2=i;
                break;
            }
        }

        int w = old_nnz[ind1]>old_nnz[ind2]? old_nnz[ind1]:old_nnz[ind2];

        new_nnz[ind1]=w;
        new_nnz[ind2] = w;
        new_idx[ind1] = ng[1];
        new_idx[ind2] = ng[0];
        
        

        
        

    }

}
