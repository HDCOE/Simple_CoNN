#include <cassert>
#include <vector>

#include <cstdint>
#include <iostream>
#include <fstream>

#include <string.h>
#include <stdio.h>
#include <stdlib.h> 
#include <math.h>
#include <thread>

#define mini_batch 16

#include "Tensor.h"


template<typename T,int x,int y ,int z>
struct  test
{
	T * ptr;

	T _data[x * y * z];

	point_t size;

	test()
	{
		ptr = _data;

		size.x = x; size.y = y; size.z = z;
	}

	T& operator()(int _x , int _y, int _z)
	{
		return _data[_z *( x * y) + _y * (x) + _x];//return this->get(_x, _y, _z);
	}

	T* data()
	{	
		return _data;
	}

};


void print_t(float * tensor, point_t size )
{
	printf("dimension %d x %d x %d \n", size.x, size.y, size.z );

	for ( int mz = 0; mz < size.z ; mz++ )
	{
		printf("-----------------------------\n");

			for ( int my = 0; my < size.y ; my++ )
			{
				for ( int mx = 0; mx < size.x ; mx++ )
				{
					std:: cout << " "<< tensor[mz *(size.x * size.y) + my * size.x + mx];//printf( "%.3f \t", tensor(mx,my,mz));
				}
				printf( "\n" );
			}
	}
	printf("-------------------------\n");
}

int main()
{

	tensor_t <float,2,2,1>  t1;
	t1(0,0,0) = 1;
	t1(1,0,0) = 2;
	t1(0,1,0) =3;
	t1(1,1,0) = 4;

	tensor_t <float,2,2,1> t2;
	t2 = t1;
	t2(0,0,0) = 5;

	print_tensor(t2.data() ,t2.size);

	print_tensor(t1.data() ,t1.size);

	return 0;
}