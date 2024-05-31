#ifndef CSR
#define CSR

#include <cstddef>
#include <iostream>
using namespace std;

typedef unsigned int Vertex;
typedef unsigned long Weight;

template <class T> void Print_array(const T *array, const int size)
{
    for (int i = 0; i < size; i++)
    {
        cout << array[i] << " ";
    }
    cout << endl;
}

template <class T> bool compareArr(T *a, T *b, Vertex size)
{
    bool out = true;
    for (int i = 0; i < size; i++)
    {
        if (a[i] != b[i])
        {
            out = false;
            break;
        }
    }

    return out;
}
struct CSR_mat
{
    Weight *nnz;
    Vertex *idx;
    Vertex *cct;

    Vertex vertex_count;
    Vertex edge_count;

    bool OnCPU;

    Vertex Get_Vertex_count() const { return vertex_count; }
    Vertex Get_edge_count() const { return edge_count; }

    void Allocate(const Vertex n_vertices, const Vertex n_edges,
                  bool cpu = true)
    {

        vertex_count = n_vertices;
        edge_count = n_edges;
        OnCPU = cpu;
        if (cpu)
        {

            nnz = (Weight *)malloc(sizeof(Weight) * 2 * n_edges);
            idx = (Vertex *)malloc(sizeof(Vertex) * 2 * n_edges);
            cct = (Vertex *)malloc(sizeof(Vertex) * (n_vertices + 1));
        }

        else
        {
            cudaMalloc(&nnz, sizeof(Weight) * 2 * n_edges);
            cudaMalloc(&idx, sizeof(Vertex) * 2 * n_edges);
            cudaMalloc(&cct, sizeof(Vertex) * (n_vertices + 1));
        }
    }

    void Deallocate()
    {
        if (OnCPU)
        {
            if (nnz != nullptr)
            {
                free(nnz);
            }
            if (idx != nullptr)
            {
                free(idx);
            }
            if (cct != nullptr)
            {
                free(cct);
            }
        }

        else
        {
            if (nnz != nullptr)
            {
                cudaFree(nnz);
            }
            if (idx != nullptr)
            {
                cudaFree(idx);
            }
            if (cct != nullptr)
            {
                cudaFree(cct);
            }
        }
    }

    void print_mat()
    {
        if (OnCPU)
        {

            Print_array(nnz, 2 * edge_count);
            cout << endl;
            Print_array(idx, 2 * edge_count);
            cout << endl;
            Print_array(cct, vertex_count + 1);
            cout << endl;
        }
        else
        {
            cout << "Matrix is allocated in GPU" << endl;
        }
    }
};

void copy_csr_mat(const CSR_mat &curr, CSR_mat &copy, bool changeHost = false)
{
    const Vertex edges = curr.Get_Vertex_count();
    const Vertex v = curr.Get_edge_count();
   
    if (!curr.OnCPU)
    {
        if (!changeHost)
        {
            copy.Allocate(curr.Get_Vertex_count(), curr.Get_edge_count(), false);

            cudaMemcpy(copy.nnz, curr.nnz, sizeof(Weight) * 2 * edges,
                       cudaMemcpyDeviceToDevice);
            cudaMemcpy(copy.idx, curr.idx, sizeof(Vertex) * 2 * edges,
                       cudaMemcpyDeviceToDevice);
            cudaMemcpy(copy.cct, curr.cct, sizeof(Vertex) * (v + 1),
                       cudaMemcpyDeviceToDevice);
        }
        else
        {
            copy.Allocate(curr.Get_Vertex_count(), curr.Get_edge_count(), true);

            cudaMemcpy(copy.nnz, curr.nnz, sizeof(Weight) * 2 * edges,
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(copy.idx, curr.idx, sizeof(Vertex) * 2 * edges,
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(copy.cct, curr.cct, sizeof(Vertex) * (v + 1),
                       cudaMemcpyDeviceToHost);
        }
    }
    else
    {
        if (!changeHost)
        {
            copy.Allocate(curr.Get_Vertex_count(), curr.Get_edge_count(), true);

            memcpy(copy.nnz, curr.nnz, sizeof(Weight) * 2 * edges);
            memcpy(copy.idx, curr.idx, sizeof(Vertex) * 2 * edges);
            memcpy(copy.cct, curr.cct, sizeof(Vertex) * (v + 1));
        }
        else
        {
            copy.Allocate(curr.Get_Vertex_count(), curr.Get_edge_count(), false);

            cudaMemcpy(copy.nnz, curr.nnz, sizeof(Weight) * 2 * edges,
                       cudaMemcpyHostToDevice);
            cudaMemcpy(copy.idx, curr.idx, sizeof(Vertex) * 2 * edges,
                       cudaMemcpyHostToDevice);
            cudaMemcpy(copy.cct, curr.cct, sizeof(Vertex) * (v + 1),
                       cudaMemcpyHostToDevice);
        }
    }
}

#endif