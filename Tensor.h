using namespace  std;

enum layer_type
{
	conv, pooling, fc, relu, softmax
};

struct point_t
{
	int x, y, z;
};


template<typename T, int x, int y, int z>
struct tensor_t
{

	T _data[x * y * z];

	point_t size;

/*Define size of tensor */
	tensor_t()
	{
		size.x = x;
		size.y = y;
		size.z = z;
		// initialize to be 0
		for (int i = 0; i < x * y * z; ++i)
		{
			this->_data[i] = 0;
		}
	}

/* Define element of tensor by tensor(x,y,z)= value */
	T& operator()( int _x, int _y, int _z )
	{
		return _data[_z *( x * y) + _y * (x) + _x];
	}

	T* data()
	{	
		return _data;
	}

	/*void fill(T value)
	{
		for (int i = 0; i < (x * y * z); ++i)
		{
			_data[i] = value;
		}
	}*/

};

/*Print all data in tensor*/
static void print_tensor( float * tensor, point_t size )
{
	printf("dimension %d x %d x %d \n", size.x, size.y, size.z );
	for ( int mz = 0; mz < size.z ; mz++ )
	{
		printf("-----------------------------\n");//printf( "[Dim%d]\n", z );
		for ( int my = 0; my < size.y ; my++ )
		{
			for ( int mx = 0; mx < size.x ; mx++ )
			{
				printf( "%.3f \t", tensor[mz *(size.x * size.y) + my * size.x + mx]);
			}
			printf( "\n" );
		}
	}
	printf("-------------------------\n");
}

void fill( float * t, point_t size, float value)
{	
	for (int i = 0; i < size.x * size.y * size.z; ++i)
	{
		t[i] = value;
	}	
}
