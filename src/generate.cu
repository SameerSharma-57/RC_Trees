
#include <algorithm>
#include <bits/types/FILE.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <unordered_map>

typedef unsigned int Vertex;
typedef unsigned long Weight;

using namespace std;


Vertex random_vertex(Vertex max, Vertex min=0)
{
    return (min+(Vertex)(rand() / (double)(RAND_MAX) * (max-min)));
}

Weight random_weight(Weight w)
{
    return 1 + (Weight)(rand() / (double)(RAND_MAX) * (w));
}


void SaveRandomTreeToFile(Vertex v,string path){
    FILE *ptr;
    Vertex edges = v-1;
    const char *c_path=path.c_str();
    ptr=fopen(c_path,"w");
    fprintf(ptr,"%d %d\n",v,edges);
    Vertex v1;
    Vertex v2=1;
    Weight w;


    while(edges--){
        v1=random_vertex(v2-1);
        w=random_weight(1000);
        fprintf(ptr,"%d %d %lu\n",v1,v2,w);
        v2++;
    }

    fclose(ptr);
    
}   

void SaveRandomGraphToFile(Vertex v, Vertex edges, string path)
{
    FILE *ptr;
    const char *c_path = path.c_str();
    ptr = fopen(c_path, "w");
    fprintf(ptr, "%d %d\n", v, edges);
    Vertex v1, v2;
    Weight w;
    int arr[v];
    for (int i = 0; i < v; i++)
    {
        arr[i] = i;
    }
    // unsigned seed =
    // std::chrono::system_clock::now().time_since_epoch().count();
    random_shuffle(arr, arr + v);
    unordered_map<Vertex, bool> m;
    int temp = 0;
    while (temp < v && edges-- > 0)
    {
        if (temp == 0)
        {
            v2 = arr[v - 1];
        }
        else
        {
            v2 = arr[temp - 1];
        }
        v1 = arr[temp];
        w = random_weight(1000);
        m[v1 * v + v2] = 1;
        m[v2 * v + v1] = 1;
        fprintf(ptr, "%d %d %lu\n", v1, v2, w);
        temp++;
    }

    while (edges)
    {
        v1 = random_vertex(v);
        v2 = random_vertex(v);
        if (v1 != v2 && m.find(v1 * v + v2) == m.end() &&
            m.find(v2 * v + v1) == m.end())
        {
            edges--;
            w = random_weight(1000);
            m[v1 * v + v2] = 1;
            m[v2 * v + v1] = 1;
            fprintf(ptr, "%d %d %lu\n", v1, v2, w);
        }
    }
    // fprintf(ptr,"Hello World");

    fclose(ptr);
}



#include "CSR_matrix.cu"

struct Edge
{
    Vertex v1;
    Vertex v2;
    Weight w;
};

bool CompareEdge(const Edge a, const Edge b) { return a.v1!=b.v1? a.v1 < b.v1 : a.v2 < b.v2; }

CSR_mat ReadSparseGraph(const string path, const bool directed)
{
    FILE *ptr;
    const char *c_path = path.c_str();
    ptr = fopen(c_path, "r");
    Vertex n_vertices;
    Vertex n_edges;

    fscanf(ptr, "%d %d", &n_vertices, &n_edges);

    CSR_mat g;
    g.Allocate(n_vertices, n_edges);
    Vertex v1;
    Vertex v2;
    Weight w;
    Edge *arr=(Edge*)malloc(sizeof(Edge)*(2*n_edges));
    Vertex ind = 0;
    while (ind < 2 * n_edges)
    {
        fscanf(ptr, "%d %d %lu", &v1, &v2, &w);
        arr[ind].v1 = v1;
        arr[ind].v2 = v2;
        arr[ind].w = w;

        ind++;

        arr[ind].v1 = v2;
        arr[ind].v2 = v1;
        arr[ind].w = w;

        ind++;
    }

    sort(arr, arr + (2 * n_edges), CompareEdge);

    for (int i = 0; i < 2 * n_edges; i++)
    {
        g.nnz[i] = arr[i].w;
        g.idx[i] = arr[i].v2;
    }

    Vertex cnt;
    ind = 0;
    g.cct[0] = 0;
    g.cct[n_vertices] = n_edges * 2;
    for (int i = 0, j = 0; i < n_vertices; i++)
    {
        cnt = 0;
        while (arr[j].v1 == i)
        {
            cnt++;
            j++;
        }
        g.cct[i + 1] = g.cct[i] + cnt;
    }
    return g;
}
