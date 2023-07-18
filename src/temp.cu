#include <cstdio>

__global__ void myfunc(bool *l){
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if(tid<5){
        l[tid]=1;
    }
}

int main(){


    bool *l = (bool* )malloc(sizeof(bool)*5);

    memset(l,0,sizeof(bool)*5);

    bool *l_gpu;
    cudaMalloc(&l_gpu,sizeof(bool)*5);
    cudaMemset(l_gpu, 0, sizeof(bool)*5);
    myfunc<<<10,1>>>(l_gpu);
    cudaMemcpy(l, l_gpu, sizeof(bool)*5, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 5; i++)
    {
        printf("%d ",l[i]);
    }
    

    return 0;
}