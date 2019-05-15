using namespace std;

void Padding(float * out, point_t out_size, float * in, int padsize)
{
	int idx = 0;
	for (int z = 0; z < out_size.z; ++z)
		for (int y = 0; y < out_size.y; ++y)
			for (int x = 0; x < out_size.x; ++x)
			{
				if( (x >= padsize && y >= padsize) && (x < out_size.x-padsize && y < out_size.y - padsize))
				{
					out[z *( out_size.x * out_size.y) + y * out_size.x + x] = in[idx];
					idx ++;
				}
				else
					out[z *( out_size.x * out_size.y) + y * out_size.x + x] = 0;			
			}
}

template<int Wsize, int Nfilter, int stride, int pad, int in_x, int in_y, int in_z>
struct  conv_layer 
{
	layer_type type = conv;
	tensor_t<float, in_x, in_y, in_z> input;

	tensor_t<float, (in_x - Wsize + 2 * pad)/stride + 1, 
				    (in_y - Wsize + 2 * pad)/stride + 1,
	 				 Nfilter > output, dZ;

	tensor_t<float, in_x, in_y, in_z> gradient_dA;

	tensor_t<float, Wsize, Wsize, in_z> W[Nfilter];
	tensor_t<float, 1, 1, Nfilter> bias;

	
	tensor_t<float, in_x + (2 * pad), in_y + (2 * pad), in_z> inpad, dA_padd; 

	tensor_t<float, Wsize, Wsize, in_z> gradient_dW[Nfilter];
	tensor_t<float, 1, 1, Nfilter> gradient_dB;


	//tensor_t<float, in_x + (2 * pad), in_y + (2 * pad), in_z> dA_padd;
// momentum 
	tensor_t<float, Wsize, Wsize, in_z> v_dw[Nfilter], s_dw[Nfilter];
	tensor_t<float, 1, 1, Nfilter> v_db, s_db;

	int vert_start, vert_end, horiz_start, horiz_end;

//relu
	relu_layer<(in_x - Wsize + 2 * pad)/stride + 1, 
			   (in_y - Wsize + 2 * pad)/stride + 1,
	 		    Nfilter> relu1;

	conv_layer()
	{
		// initial
		//fill(bias.data(), bias.size, 0);

		//fill(gradient_dB.data(), gradient_dB.size,0);
		//fill(gradient_dA.data(), gradient_dA.size, 0);
		//fill(dA_padd.data(), dA_padd.size, 0);
		//fill(v_db.data(), v_db.size, 0);

		// init weight
		for (int i = 0; i < Nfilter; ++i)
		{
		// constant
			//fill(W[i].data(), W[i].size, w_value);

		// init with xavier
			initweight(W[i].data(), Wsize, input.size, output.size);
		}

			// init bias
			//fill(bias.data(),bias.size, 0.5);
		
	}

	void initweight_const(float w_value)
	{
		for (int i = 0; i < Nfilter; ++i)
		{
		// constant
			fill(W[i].data(), W[i].size, w_value );
		}
	}

	void forward() // forward with relu
	{
		this->forward_conv();
		relu1.input = this->output;
		relu1.forward();
		this->output = relu1.output;

	}
	void forward_conv()
	{
		//print_tensor(input.data(),input.size);
		Padding(inpad.data(), inpad.size, input.data(),pad);

		for (int idx_filter = 0; idx_filter < Nfilter; ++idx_filter)
		{
			for (int idx_y = 0; idx_y < output.size.y ; ++idx_y)
			{
				for (int idx_x = 0; idx_x < output.size.x; ++idx_x)
				{
					float output_c = 0;
					vert_start = idx_y * stride;
					horiz_start = idx_x * stride;

					for (int ch = 0; ch < input.size.z; ++ch)
					{
						//convolution
						for (int v = vert_start; v < vert_start + Wsize; ++v)
							for (int ho = horiz_start; ho < horiz_start + Wsize; ++ho)
							{
								output_c += inpad(v,ho,ch) * W[idx_filter](v - vert_start, ho - horiz_start, ch );
							}
						// end conv
					}// end conv ch
					this->output(idx_y, idx_x, idx_filter) = output_c + bias(0,0,idx_filter);
				}
			}
		}
	} // end forward_conv

	void backward()
	{
		relu1.dZ = dZ;
		relu1.backward();
		
		this->dZ = relu1.gradient_dA;
		this->backward_conv();
	}

	void backward_conv()
	{
		fill(dA_padd.data(), dA_padd.size, 0);

		float W_current = 0.0;
		float grad_db = 0.0;
		float dZ_cache;

		for (int idx_y = 0; idx_y < dZ.size.y; ++idx_y)
		{
			for (int idx_x = 0; idx_x < dZ.size.x; ++idx_x)
			{
				vert_start = idx_y;
				horiz_start = idx_x;

				for (int idx_filter = 0; idx_filter < Nfilter; ++idx_filter)
				{
					dZ_cache = dZ(idx_y, idx_x, idx_filter);

					for (int ch = 0; ch < input.size.z ; ++ch)
					{
						for (int v = vert_start; v < vert_start + Wsize; ++v)
							for (int ho = horiz_start; ho < horiz_start + Wsize; ++ho)
							{
							
						// Calculate dA by dA += sum(W*dZ[h,w])
								W_current = W[idx_filter](v - vert_start, ho - horiz_start, ch); //4.5 - 3.6 = 0.9
								float& dA_padd_ = dA_padd(v,ho,ch);
								dA_padd_ +=  W_current * dZ_cache; //5.5 -3.6 = 1.9
							
						// calculate dW += sum (a_slice*dZ[h,w])
								gradient_dW[idx_filter](v - vert_start, ho - horiz_start,ch) +=  inpad(v,ho,ch) * dZ_cache;
							}
						// end loop
					}
				}
			}
		}// end loop

		// calculate db = sum (dZ)
		for (int idx_filter = 0; idx_filter < Nfilter; ++idx_filter)
		{
			grad_db = 0;

			for (int i_y = 0; i_y < dZ.size.y; ++i_y)
				for (int i_x = 0; i_x < dZ.size.x; ++i_x)
					grad_db += dZ(i_y, i_x, idx_filter);

			gradient_dB(0,0,idx_filter) = grad_db;	
		}// end loop

		// copy from da_padd but cut the pad
		for (int ch = 0; ch < input.size.z; ++ch)
		 {
		 	for (int i_x = 0; i_x < gradient_dA.size.x; ++i_x)
		 	{
		 		for (int i_y = 0; i_y < gradient_dA.size.y; ++i_y)
		 		{
		 			gradient_dA(i_x, i_y, ch) = dA_padd(i_x + pad, i_y + pad, ch);
		 		}
		 	}
		 } 
	}// end backward

	void weight_update()
	{
		for (int i_filter = 0; i_filter < Nfilter; ++i_filter)
		{
			for (int i_x = 0; i_x < Wsize; ++i_x)
				for (int i_y = 0; i_y < Wsize; ++i_y)
					for (int ch = 0; ch < input.size.z; ++ch)
					{
					// update weights
						float& w 	  = W[i_filter](i_x, i_y, ch);
						float& grad   = gradient_dW[i_filter](i_x, i_y, ch);
						float& v_dw_t = v_dw[i_filter](i_x, i_y, ch);
						// adam 
						float& s_dw_t = s_dw[i_filter](i_x, i_y, ch);
						Adam(w, grad, v_dw_t, s_dw_t);//SGD_update(w, grad, v_dw_t );	
					}
			// end loop w

			float& b = bias(0,0,i_filter);
			float& b_grad = gradient_dB(0,0,i_filter);
			float& v_db_t = v_db(0,0,i_filter);

			//adadm
			float& s_db_t = s_db(0,0,i_filter);
			Adam(b,b_grad, v_db_t, s_db_t); //SGD_update(b,b_grad, v_db_t);

		}
		
	} // end weight update
};
