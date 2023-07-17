#ifndef BFS
#define BFS
#include "CSR_matrix.cu"
#include "graph.cu"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <queue>

struct BFS_result
{
    Vertex vertex_count;
    Weight *dist;
    Vertex *pred;

    bool OnCPU;

    void Allocate(const Vertex n, bool cpu)
    {
        vertex_count = n;
        this->OnCPU = cpu;
        if (OnCPU)
        {
            dist = (Weight *)malloc(sizeof(Weight) * vertex_count);
            pred = (Vertex *)malloc(sizeof(Vertex) * vertex_count);
        }

        else
        {
            cudaMalloc(&dist, sizeof(Weight) * vertex_count);
            cudaMalloc(&pred, sizeof(Vertex) * vertex_count);
            cudaMemset(dist, 0xff, sizeof(Weight) * vertex_count);
        }
    }

    void Deallocate()
    {
        if (dist != nullptr)
        {

            if (OnCPU)
            {
                free(dist);
            }
            else
            {
                cudaFree(dist);
            }
        }

        if (pred != nullptr)
        {

            if (OnCPU)
            {
                free(pred);
            }

            else
            {
                cudaFree(pred);
            }
        }
    }

    void Print_result()
    {
        cout << "\nParallel Implementation\nDist : ";
        Print_array(dist, vertex_count);
        cout << "\nPred : ";
        Print_array(pred, vertex_count);
    }
};

bool Get_Bfs(const Graph &g, const Vertex source, const Vertex destination,
             BFS_result &result)
{
    Vertex vertex_count = g.Get_Vertex_count();
    if (source >= vertex_count)
    {
        return false;
    }

    queue<Vertex> q;
    q.push(source);
    for (Vertex i = 0; i < vertex_count; i++)
    {
        result.dist[i] = 0;
    }
    result.dist[source] = 0;
    result.pred[source] = source;

    Vertex v;
    while (!q.empty())
    {
        v = q.front();
        q.pop();
        for (int i = 0; i < vertex_count; i++)
        {
            if (g.Get_Edge(v, i) == 0 || result.dist[i] != 0 || i == source)
                continue;
            q.push(i);
            result.dist[i] = result.dist[v] + 1;
            result.pred[i] = v;
            if (i == destination)
            {
                break;
            }
        }
    }

    return true;
}

bool Get_Bfs(const CSR_mat &g, const Vertex source, const Vertex destination,
             BFS_result &result)
{
    const Vertex max = UINT32_MAX;
    Vertex vertex_count = g.Get_Vertex_count();
    if (source >= vertex_count)
    {
        return false;
    }

    queue<Vertex> q;
    q.push(source);
    for (Vertex i = 0; i < vertex_count; i++)
    {
        result.dist[i] = max;
    }
    result.dist[source] = 0;
    result.pred[source] = source;

    Vertex v;
    while (!q.empty())
    {
        v = q.front();
        q.pop();
        Vertex neighbor;
        for (int i = g.cct[v], j = g.cct[v+1]; i < j; i++)
        {
            neighbor = g.idx[i];
            if (result.dist[neighbor] != max)
                continue;
            q.push(neighbor);
            result.dist[neighbor] = result.dist[v] + 1;
            result.pred[neighbor] = v;
            if (neighbor == destination)
            {
                break;
            }
        }
    }

    return true;
}

__device__ const Weight unvisited = 0xffffffffffffffff;

BFS_result Get_Bfs(const Graph &g, const Vertex source,
                   const Vertex destination)
{
    BFS_result result;
    result.Allocate(g.Get_Vertex_count(), true);
    if (Get_Bfs(g, source, destination, result) == false)
    {
        throw result;
    }
    return result;
}

BFS_result Get_Bfs(const CSR_mat &g, const Vertex source,
                   const Vertex destination)
{
    BFS_result result;
    result.Allocate(g.Get_Vertex_count(), true);
    if (Get_Bfs(g, source, destination, result) == false)
    {
        throw result;
    }
    return result;
}

#include <cstdio>

__global__ void Bfs_kernel(const Weight *adjmat, Vertex vertex_count,
                           BFS_result result, Vertex level, bool *update)
{
    unsigned const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= vertex_count || result.dist[tid] != unvisited)
    {
        return;
    }

    for (int i = 0; i < vertex_count; i++)
    {
        bool r1 = adjmat[(vertex_count * i) + tid];
        // printf("TID %d stage 1, edge exists: %d \n", tid, r1);
        if (i != tid && r1 && result.dist[i] == level)
        {
            result.dist[tid] = result.dist[i] + 1;
            result.pred[tid] = i;
            *update = true;
            break;
        }
    }
}

BFS_result Get_Bfs_parallel(Graph &g, const Vertex source,
                            const Vertex destination)
{
    BFS_result result, host_result;
    const Vertex vertex_count = g.Get_Vertex_count();
    result.Allocate(vertex_count, false);
    host_result.Allocate(vertex_count, true);

    bool *update;
    cudaMallocManaged(&update, sizeof(bool));
    *update = true;
    Vertex p_bytes = vertex_count * sizeof(Vertex),
           d_bytes = vertex_count * sizeof(Weight);

    cudaMemset(result.dist + source, 0, sizeof(Weight));
    host_result.pred[source] = source;
    cudaMemcpy(result.pred + source, host_result.pred + source, sizeof(Weight),
               cudaMemcpyHostToDevice);
    Vertex level = 0;
    while (*update == true)
    {
        *update = false;
        Bfs_kernel<<<((vertex_count + 1023) / 1024), 1024>>>(
            g.adjmat.Array, vertex_count, result, level, update);
        level++;
        cudaDeviceSynchronize();
    }

    cudaMemcpy(host_result.dist, result.dist, d_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_result.pred, result.pred, p_bytes, cudaMemcpyDeviceToHost);
    result.Deallocate();
    return host_result;
}
__global__ void Bfs_kernel(const Vertex* cct,const Vertex* idx, const Vertex vertex_count,
                           BFS_result result, const Vertex level, bool *update)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    Vertex neighbor;
    // printf("Stage 0 %d\n",tid);
    if (tid >= vertex_count || result.dist[tid] != UINT32_MAX)
    {
        return;
    }
    // printf("Stage 1 %d %d %d %d\n",tid,level,cct[tid],cct[tid+1]);
    // printf("%d %d\n",cct[tid],cct[tid+1]);
    for (int i = cct[tid]; i < cct[tid+1]; i++)
    {
        neighbor=idx[i];
        // printf("Stage 2 %d %d %d\n",tid,neighbor,level);
        // printf("TID %d stage 1, edge exists: %d \n", tid, r1);
        if (neighbor != tid && result.dist[neighbor] == level)
        {
            result.dist[tid] = result.dist[neighbor] + 1;
            result.pred[tid] = neighbor;
            *update = true;
            break;
        }
    }
}

BFS_result Get_Bfs_parallel(const CSR_mat &g, const Vertex source,
                            const Vertex destination)
{
    const Vertex max = UINT32_MAX;
    BFS_result result, host_result;
    const Vertex vertex_count = g.Get_Vertex_count();
    result.Allocate(vertex_count, false);
    host_result.Allocate(vertex_count, true);


    for (Vertex i = 0; i < vertex_count; i++)
    {
        host_result.dist[i] = max;
    }
    host_result.dist[source] = 0;
    host_result.pred[source] = source;
    bool *update;
    cudaMallocManaged(&update, sizeof(bool));
    *update = true;
    Vertex p_bytes = vertex_count * sizeof(Vertex),
           d_bytes = vertex_count * sizeof(Weight);


   
    cudaMemcpy(result.pred, host_result.pred, p_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(result.dist, host_result.dist, d_bytes,
cudaMemcpyHostToDevice);
    Vertex level = 0;
    while (*update == true)
    {
        *update = false;
        Bfs_kernel<<<((vertex_count + 1023) / 1024), 1024>>>(
            g.cct,g.idx, vertex_count, result, level, update);
        level++;
        cudaDeviceSynchronize();
    }

    cudaMemcpy(host_result.dist, result.dist, d_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_result.pred, result.pred, p_bytes, cudaMemcpyDeviceToHost);
    result.Deallocate();
    return host_result;
}

#endif

