
#pragma pack(push,1)
template<int x, int y, int z>
struct  relu_layer
{
	layer_type type = relu;
	tensor_t<float, x, y, z> input;
	tensor_t<float, x, y, z> output;
	tensor_t<float, x, y, z> gradient_dA;
	tensor_t<float, x, y, z> dZ;

	relu_layer()
	{

	}

	void forward()
	{
		float temp;
		for (int ix = 0; ix < x; ++ix)
			for (int iy = 0; iy < y; ++iy)
				for (int iz = 0; iz < z; ++iz)
				{
					temp = input(ix, iy, iz);

					if(temp < 0)
						output(ix, iy, iz) = 0;
					else
						output(ix, iy, iz) = temp;		
				}
		//end loop
	}

	void backward()
	{
		float temp;
		for (int ix = 0; ix < x; ++ix)
			for (int iy = 0; iy < y; ++iy)
				for (int iz = 0; iz < z; ++iz)
				{
					temp = input(ix, iy, iz);
					if(temp <= 0)
						gradient_dA(ix, iy, iz) = 0;
					else
						gradient_dA(ix, iy, iz) = dZ(ix, iy, iz);
				}
		// end loop

	}
};


#pragma pack(pop)

