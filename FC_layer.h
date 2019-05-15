
/* we can do conv of x with  w which has same size as x,
 or we can stretch x to be vector and multiply with vector w */
#pragma pack(push,1)

template<int n_class, int x, int y, int z>
struct fc_layer
{
	layer_type type = fc;
	tensor_t<float, x, y, z > input;
	tensor_t<float, 1, 1, n_class > output;
	tensor_t<float, x, y, z > gradient_dA;
	
	tensor_t<float, x, y, z> W[n_class], v_dw[n_class], s_dw[n_class]; //std::vector<tensor_t<float>> W;
	tensor_t<float, 1, 1, n_class> bias, v_db, s_db;
	
	//cache gradient;
	tensor_t<float, x, y, z> gradient_dW[n_class];
	tensor_t<float, 1, 1, n_class> gradient_dB;

	tensor_t<float, 1, 1, n_class> dZ;

	 // momentum
	//std::vector<tensor_t<float>> v_dw;
	//tensor_t<float> v_db;

	fc_layer()
	{
		//fill(bias.data(), bias.size, 0);
		//fill(gradient_dB.data(), gradient_dB.size, 0);

		for (int i = 0; i < n_class; ++i)
		{
			initweight(W[i].data(), W[i].size.x, input.size, output.size);
			
			//initweight(v_dw[i].data(),  v_dw[i].size.x, input.size, output.size);

			//fill(gradient_dW[i].data(), gradient_dW[0].size, 0);
		}

	}
	
		void initweight_const(float w_value)
	{
		for (int i = 0; i < n_class; ++i)
		{
		// constant
			fill(W[i].data(), W[i].size, w_value);
		}
	}
	void forward()
	{
		float dot_out = 0;

		for (int i_filter = 0; i_filter < n_class; ++i_filter)
		{
			dot_out = 0;
			// dot w . in
			for (int i_x = 0; i_x < x; ++i_x)
				for (int i_y = 0; i_y < y; ++i_y)
					for (int i_z = 0; i_z < z; ++i_z)
					{
						dot_out += input(i_x, i_y, i_z) * W[i_filter](i_x, i_y, i_z);
					}

			output(0,0, i_filter) = dot_out;
		} // end loop filter
	} // end forward

	void backward()
	{
		fill(gradient_dA.data(), gradient_dA.size, 0);
		float dZ_cache;

		for (int i_filter = 0; i_filter < n_class; ++i_filter)
		{
			dZ_cache = dZ(0,0,i_filter);

			for (int i_x = 0; i_x < x; ++i_x)
				for (int i_y = 0; i_y < y; ++i_y)
					for (int i_z = 0; i_z < z; ++i_z)
					{
						gradient_dW[i_filter](i_x, i_y, i_z) += dZ_cache * input(i_x, i_y, i_z);
						gradient_dA(i_x, i_y, i_z) += W[i_filter](i_x, i_y, i_z) * dZ_cache;
					
					}
			// end loop

			gradient_dB(0,0,i_filter) += dZ_cache;
		}


	} // end backward

	void weight_update()
	{
		for (int i_filter = 0; i_filter < n_class; ++i_filter)
		{
			for (int i_x = 0; i_x < x; ++i_x)
				for (int i_y = 0; i_y < y; ++i_y)
					for (int i_z = 0; i_z < z; ++i_z)
					{
						float& w    = W[i_filter](i_x, i_y, i_z);
						float& grad = gradient_dW[i_filter](i_x, i_y, i_z);
						float& v_dw_t = v_dw[i_filter](i_x, i_y, i_z);
						//adam
						float& s_dw_t = s_dw[i_filter](i_x, i_y, i_z);
						Adam(w, grad, v_dw_t, s_dw_t);//SGD_update(w, grad, v_dw_t );
					}
			//end loop w
			// update bias 
			float& b = bias(0,0,i_filter);
			float& b_grad = gradient_dB(0,0,i_filter);
			float& v_db_t = v_db(0,0,i_filter);
			//adam
			float& s_db_t = s_db(0,0,i_filter);
			Adam(b,b_grad, v_db_t, s_db_t); //SGD_update(b,b_grad, v_db_t);

		}

	}
};
#pragma pack(pop)