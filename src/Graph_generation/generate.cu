#include "Square_Matrix.cu"
#include "graph.cu"

#include <algorithm>
#include <bits/types/FILE.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <unordered_map>

Vertex random_vertex(Vertex max)
{
    return (Vertex)(rand() / (double)(RAND_MAX) * (max));
}

Weight random_weight(Weight w)
{
    return 1 + (Weight)(rand() / (double)(RAND_MAX) * (w));
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

Graph ReadGraphFromFile(string path, bool& success, bool isDirected = true)
{
    success=true;
    FILE *ptr;
    const char *c_path = path.c_str();
    ptr = fopen(c_path, "r");
    // if(ptr==NULL){
    //     printf("Path does not exist");
    //     return NULL;
    // }
    Vertex v, e;
    fscanf(ptr, "%d %d", &v, &e);
    // printf("%d %d",v,e);
    Graph g;
    if(!g.Allocate(v, true)){
        success=false;
        return g;
    }
    Vertex v1, v2;
    Weight w=1;
    while (e>0)
    {
        fscanf(ptr, "%d %d", &v1, &v2);
        // printf("%d %d %lu",v1,v2,w);
        g.Set_Edge(v1, v2, w);
        if (!isDirected)
        {
            g.Set_Edge(v2, v1, w);
        }
        e--;
    }
    return g;
}


