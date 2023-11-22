

#include "CSR_matrix.cu"
#include <cstring>
#include <omp.h>

void getLeaf(const CSR_mat g,bool* l,Vertex* d){

    #pragma omp parallel for
    for (int i = 0; i < g.Get_Vertex_count(); i++)
    {
        int n_neighbours=0;
        if(d[i]==0){
            n_neighbours=0;
            for (int j = g.cct[i]; j < g.cct[i+1]; j++)
            {
                if(d[g.idx[j]]==0){
                    n_neighbours++;
                    if(n_neighbours>1){
                        break;
                    }
                }
            }
            l[i] = (n_neighbours==1);
        }
        
    }
    
}


void Compute(const Vertex round,const CSR_mat g,const bool*l,Vertex*d, bool*update){
    #pragma omp parallel for
    for (int i = 0; i < g.Get_Vertex_count(); i++)
    {
        int n_neighbours=0;
        if(d[i]==0){
            n_neighbours=0;
            for (int j = g.cct[i]; j < g.cct[i+1]; j++)
            {
                if(d[g.idx[j]]==0){


                    n_neighbours++;
                    if(l[g.idx[j]]){

                        d[g.idx[j]]=round;
                        *update=true;
                        break;
                    }
                }
            }

            if (n_neighbours==0)
            {
                d[i]=round;
                *update=true;
            }
            
            
        }
    }
    
}


Vertex* generateCompressedGraph(const CSR_mat g){

    Vertex vertices = g.Get_Vertex_count();
    bool* l = new bool[vertices];
    Vertex* d = new Vertex[vertices];
    std::memset(d,0,sizeof(Vertex)*vertices);

    bool *update = new bool;
    *update=true;

    Vertex round=1;
    while(*update){
        
        *update=false;
        getLeaf(g, l, d);
        // Print_array(d,vertices);
        // Print_array(l, vertices);
        Compute(round, g, l, d, update);
        round++;

    }


    return d;

}