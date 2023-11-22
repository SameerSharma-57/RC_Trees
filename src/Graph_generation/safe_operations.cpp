#include <cstdint>
typedef unsigned int Vertex;
typedef unsigned long Weight;

Vertex Safe_add_int(Vertex a,Vertex b){
    if(UINT32_MAX-a<b){
        return 0;
    }
    return a+b;
}

Vertex Safe_mult_int(Vertex a,Vertex b){
    if(UINT32_MAX/a<b){
        return 0;
    }
    return a*b;
}

Weight Safe_add_long(Weight a,Weight b){
    if(UINT64_MAX-a<b){
        return 0;
    }
    return a+b;
}

Weight Safe_mult_long(Weight a,Weight b){
    if(UINT64_MAX/a<b){
        return 0;
    }
    return a*b;
}