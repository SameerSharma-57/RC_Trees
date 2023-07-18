#include <cstdio>   

#include "rake.cu"

#include "generate.cu"

template<class T>
void printArr(T* arr,Vertex size){
    for (int i = 0; i < size; i++)
    {
        cout<<arr[i]<<" ";
    }
    cout<<endl;
    
}


int main(){


    string input_file = "/home/sameer/RC_trees/Graphs/graph_2.txt";
    freopen("/home/sameer/RC_trees/Graphs/graph_output.txt","w",stdout);
   
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


    printArr(g_sparse.nnz,2*edges);

    Vertex *d;
    d = GenerateCompressedGraph(g_parallel_sparse);

    printArr(d,vertices);
    

    return 0;

}