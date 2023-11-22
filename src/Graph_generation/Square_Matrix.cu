#ifndef SQMT
#define SQMT

#include <cstddef>
#include "safe_operations.cpp"
typedef unsigned int Vertex;
typedef unsigned long Weight;
#include <iostream>
using namespace std;
struct Square_matrix
{
	Weight* Array;
	int	 S;
	bool OnHost;

	bool Allocate(const int n, const bool oh = true)
	{
		OnHost = oh;
		Weight n_bytes=Safe_mult_long(2,Safe_mult_long(n,n));

		if(n_bytes==0){
			return false;
		}
		if(oh)
			Array = (Weight*)malloc(n_bytes);
		else
			cudaMalloc(&Array, n_bytes);

		if(Array==nullptr){
			return false;
		}
		S = n;
		return true;
	}

	__device__ __host__ Weight Get(const int i, const int j) const
	{
		int index = S * i + j;
		return Array[index];
	}

	__device__ __host__ void Set(const int i, const int j, const Weight value)
	{
		int index	 = S * i + j;
		Array[index] = value;
	}

	void Deallocate()
	{
		if(OnHost && Array != nullptr)
			free(Array);
		else if(OnHost == false && Array != nullptr)
			cudaFree(Array);
	}

	void print_matrix() const
	{
		for(int i = 0; i < S; i++)
		{
			for(int j = 0; j < S; j++)
			{
				cout << Get(i, j) << " ";
			}
			cout << "\n";
		}
	}
};

#endif