#pragma pack(push,1)
/* Find the maximum value of tensor*/

//float maximum(tensor_t<float>& in);
//float average(tensor_t<float>& in);

//this layer include pool -> relu

template< int poolsize, int stride, int mode, int x, int y, int z>
struct pool_layer
{
    layer_type type = pooling;
    tensor_t<float, x, y, z> input;
    tensor_t<float,(x - poolsize )/stride + 1, (y - poolsize )/stride + 1 , z> output;
    tensor_t<float, x, y, z> gradient_dA;
    tensor_t<float, (x - poolsize )/stride + 1, (y - poolsize )/stride + 1 , z> dZ;
    tensor_t<float, poolsize, poolsize, 1> a_prev_temp;

    relu_layer<(x - poolsize )/stride + 1, (y - poolsize )/stride + 1 , z> relu1;

    int vert_start,vert_end,horiz_start,horiz_end;
    pool_layer()
    {

    }

    void forward()
    {
        this->forward_pool();
        relu1.input = this->output;
        relu1.forward();
        this->output = relu1.output;
    }

    void forward_pool()
    {
        float max = 0, avg = 0;
        float cache_in;

        for (int o_y = 0; o_y < output.size.y; ++o_y)
        {
            for (int o_x = 0; o_x < output.size.x; ++o_x)
            {
                vert_start = o_y * stride;
                horiz_start = o_x * stride;

                // pool
                for (int ch = 0; ch < output.size.z; ++ch)
                {
                    max = input(vert_start, horiz_start, ch); avg = 0;

                    //pool loop
                    for (int v = vert_start; v < vert_start + poolsize; ++v)
                        for (int h = horiz_start; h < horiz_start + poolsize; ++h)
                        {
                            cache_in = input(v,h,ch);
                            avg += cache_in;
                            //find max
                            if (cache_in > max)
                                max = cache_in;
                        }
                    // end loop pool
                    if (mode == 0) // max
                        output(o_y,o_x,ch) = max;
                    else // average
                        output(o_y,o_x,ch) = avg / (poolsize * poolsize);
                }
            }
        }
    } //end forward

    void backward()
    {
        relu1.dZ = this->dZ;
        relu1.backward();
        
        this->dZ = relu1.gradient_dA;
        this->backward_pool();
    }

    void backward_pool()
    {
        float cache_in,cache_out;
        float dZ_cache,dZ_cache_avg;
        int v_max, ho_max;

        for(int iy = 0; iy < dZ.size.y; ++iy)
            for (int ix = 0; ix < dZ.size.x; ++ix)
            {
                vert_start = iy * stride;
                horiz_start = ix * stride;

                //loop chanel
                for (int ch = 0; ch < dZ.size.z; ++ch)
                {
                    dZ_cache = dZ(iy, ix, ch);
                    dZ_cache_avg = dZ_cache / (poolsize * poolsize);
                    cache_out = output(iy, ix, ch);
                    // pool loop
                    for (int v = vert_start; v < vert_start + poolsize; ++v)
                        for (int h = horiz_start; h < horiz_start + poolsize; ++h)
                        {
                            cache_in = input(v,h,ch);

                            if(mode == 1) // average
                                gradient_dA(v,h,ch) = dZ_cache_avg;
                            else //max
                                gradient_dA(v,h,ch) = (cache_out == cache_in)? dZ_cache : 0.0;
                        }
                }// end loop chanel
            }
    }
}; // end pool
/*
float maximum(tensor_t<float>& in)
{
	float out = in(0,0,0);
	for (int i = 0; i < in.size.y; ++i)
	{
		for (int j = 0; j < in.size.x ; ++j)
		{
			if (in(i,j,0) > out)
			{
				out = in(i,j,0);
			}
		}
	}
return out;

}

float average(tensor_t<float>& in)
{
	float out = 0.0;
	float size = (float)(in.size.y * in.size.x);
	for (int i = 0; i < in.size.y; ++i)
	{
		for (int j = 0; j < in.size.x ; ++j)
		{
			out += in(i,j,0);
		}
	}
return out / size;
}
*/

#pragma pack(pop)