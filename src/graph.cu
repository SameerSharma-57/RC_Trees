#ifndef GRAPH
#define GRAPH

#include "Square_Matrix.cu"
#include <cstdlib>

struct Graph
{
    Vertex edge_count;
    Square_matrix adjmat;
    bool OnCPU;

    Vertex Get_Vertex_count() const { return adjmat.S; }

    Vertex Get_Edge_count() const { return edge_count; }

    bool Allocate(Vertex max, bool cpu)
    {
        OnCPU = cpu;
        if (cpu)
        {
            if(!adjmat.Allocate(max)){
                return false;
            }
            
            memset(adjmat.Array, 0, sizeof(Weight) * max * max);
        }

        else{
            if(!adjmat.Allocate(max,false)){
                return false;
            }
            cudaMemset(&adjmat.S, 0, sizeof(Weight)*max*max);
        }
        return true;
    }

    bool Reshape(Vertex new_vertex)
    {
        
        Weight *ptr = (Weight *)realloc(
            adjmat.Array, sizeof(Weight) * new_vertex * new_vertex);
        if (ptr == nullptr)
            return false;
        adjmat.Array = ptr;
        return true;
    }

    void Deallocate() { adjmat.Deallocate(); }

    __device__ __host__ Weight Get_Edge(const Vertex v1, const Vertex v2) const
    {
        return adjmat.Get(v1, v2);
    }

    void Set_Edge(const Vertex v1, const Vertex v2, const Weight w)
    {
        adjmat.Set(v1, v2, w);
        adjmat.Set(v2, v1, w);
        edge_count++;
    }

    void Print_graph(){
        adjmat.print_matrix();
    }
};

#endif