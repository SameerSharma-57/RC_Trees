#ifndef COMPRESS2
#define COMPRESS2
#include "bfs.cu"
#include "CSR_matrix.cu"


void compress_2(const CSR_mat &g, const Vertex source){
    BFS_result result = Get_Bfs_parallel(g,source);
    result.Print_result();
}

__global__ void find_2_degree_kernel(int *c, const Vertex vertices,
                                     Vertex *d, const Vertex *cct,
                                     const Vertex *idx, const Weight* dist,const int level,const int round)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < vertices && d[tid] == 0 && dist[tid]==level)
    {

        int n_neighbours = 0;

        for (int i = cct[tid]; i < cct[tid + 1]; i++)
        {
            if (d[idx[i]] == 0)
            {
                n_neighbours += 1;
                if (n_neighbours > 2)
                {
                    break;
                }
            }
        }

        c[tid] = (n_neighbours == 2);
        d[tid] = c[tid]*round;
    }
}

__global__ void compress(const Vertex *old_cct, const Vertex *old_idx,
                         const Weight *old_nnz, Vertex *new_cct,
                         Vertex *new_idx, Weight *new_nnz, const Vertex *d,
                         const Vertex vertices, const int *c, bool *update)
{

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < vertices && c[tid] && d[tid]==0)
    {

        int ng[2];
        int ind = 0;
        for (int i = old_cct[tid]; i < old_cct[tid + 1]; i++)
        {
            if (d[old_idx[i]] == 0)
            {
                ng[ind++] = old_idx[i];
            }
        }

        int ind1 = 0;
        int ind2 = 0;

        if(ind<2){
            return;
        }

        for (int i = old_cct[ng[0]]; i < old_cct[ng[0] + 1]; i++)
        {
            if (old_idx[i] == tid)
            {
                ind1 = i;
                break;
            }
        }

        for (int i = old_cct[ng[1]]; i < old_cct[ng[1] + 1]; i++)
        {
            if (old_idx[i] == tid)
            {
                ind2 = i;
                break;
            }
        }

        int w = old_nnz[ind1] > old_nnz[ind2] ? old_nnz[ind1] : old_nnz[ind2];

        new_nnz[ind1] = w;
        new_nnz[ind2] = w;
        new_idx[ind1] = ng[1];
        new_idx[ind2] = ng[0];
        *update = true;
    }
}


#endif