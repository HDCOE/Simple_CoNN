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
#include "weightinit.h"

#include "Optimization_method.h"
#include "RELU_layer.h"
#include "CONV_layer.h"
#include "FC_layer.h"
#include "POOLING_layer.h"
#include "Softmax.h"

#include "Read_dataset.h"
#include <random>

// add progress and processing time display
#include <boost/progress.hpp>
#include <boost/timer.hpp>

float Cross_entropy(float * y_hat, float * y, int nclass) // softmax cross entropy: E = (-1/n) * sum(y*log(y_hat) + (1-y)*log(1-y_hat))
{
	float E = 0.0;

	for (int filter = 0; filter < nclass ; ++filter)
	{
		if (y[filter] == 1)
		{
			E = y[filter] * log(y_hat[filter]); // Entropy has value only correct class
		}

//		printf("E %.3f y: %.2f  yhat:%.2f \n", E, y[filter], y_hat[filter] );
	}
	//printf("Err %f\n",E );
	return E;
}


int main(int argc, char const *argv[])
{
 	vector<case_t<32,32,1,10>> dataset = read_dataset("MNIST/train-images.idx3-ubyte","MNIST/train-labels.idx1-ubyte");
 	vector<case_t<32,32,1,10>> testset = read_dataset("MNIST/t10k-images.idx3-ubyte", "MNIST/t10k-labels.idx1-ubyte");
 	//print_tensor(dataset[0].data.data(), dataset[0].data.size);

 	conv_layer<5,6,1,0,32,32,1>  c1; // in: 32x32x1 out:28x28x6
 	pool_layer<2,2,1,28,28,6>    p2; // in: 28x28x6 out:14x14x6
 	conv_layer<5,16,1,0,14,14,6> c3; // in: 14x14x6 out:10x10x16
 	pool_layer<2,2,1,10,10,16>   p4; // in: 10x10x16 out: 5x5x16
 	conv_layer<5,120,1,0,5,5,16> c5; // in: 5x5x16 out:1x1x120
 	fc_layer<10,1,1,120> fc6;		 // in:1x1x120 out 1x1x10

 	softmax_class<10> sf_estimate;

 	float err = 0;
 	float n = 0;

 	int train_size = (int)dataset.size();
 	int test_size = (int)testset.size();

 	 	// Display
 	boost::progress_display disp1(train_size);
 	boost::timer t;


 	// init weight
 	c1.initweight_const(0.25);
 	c3.initweight_const(0.01);
 	c5.initweight_const(0.001);
 	fc6.initweight_const(0.01);

 	for (int ep = 0; ep < 10; ++ep)
 	{
 		n = 0; err = 0;
 	
 		for (int k = 0; k < train_size; k += mini_batch)
		{
			for (int i = k; i < mini_batch + k; ++i)
			{
				n++;
		
				c1.input = dataset[i].data; c1.forward();
				p2.input = c1.output; 		p2.forward();
				c3.input = p2.output;		c3.forward();
				p4.input = c3.output;		p4.forward();
				c5.input = p4.output;		c5.forward();
				fc6.input = c5.output;		fc6.forward();

				sf_estimate.input = fc6.output;  sf_estimate.forward();
				err += Cross_entropy(sf_estimate.y_hat.data(), dataset[i].out.data(), 10);

				// backward
				sf_estimate.y_out = dataset[i].out; sf_estimate.backward();
				fc6.dZ = sf_estimate.gradient_dA;	fc6.backward();
				c5.dZ  = fc6.gradient_dA;			c5.backward();
				p4.dZ  = c5.gradient_dA;			p4.backward();
				c3.dZ  = p4.gradient_dA;			c3.backward();
				p2.dZ  = c3.gradient_dA;			p2.backward();
				c1.dZ  = p2.gradient_dA;			c1.backward();

			}
			c1.weight_update();
			c3.weight_update();
			c5.weight_update();
			fc6.weight_update();
			
			
			if (((int)n/10)%100 == 0)
			{
				cout << "Train " << n << "  : this is epoch: " << ep << " error (%) = " << ((-1 / n) * err * 100)  << endl;	
			}
			//disp1 += mini_batch;

		} // end loop dataset
		
		// display processing time and error
		std::cout << endl << t.elapsed() << "s elapsed." << std::endl;
		cout << "Train " << train_size << ": this is epoch: " << ep+1 << " error (%) = " << ((-1 / n) * err * 100)  << endl;

		// test
		err = 0; n =0;
		for (int j = 0; j < test_size; ++j)
		{
			n++;
				c1.input = testset[j].data; c1.forward();
				p2.input = c1.output; 		p2.forward();
				c3.input = p2.output;		c3.forward();
				p4.input = c3.output;		p4.forward();
				c5.input = p4.output;		c5.forward();
				fc6.input = c5.output;		fc6.forward();

				sf_estimate.input = fc6.output;  sf_estimate.forward();
				err += Cross_entropy(sf_estimate.y_hat.data(), testset[j].out.data(), 10);

				//cout << "Test " << test_size << ": this is epoch: " << ep+1 << " error (%) = " << ((-1 / n) * err * 100)  << endl;
		 }

		cout << "Test " << test_size << ": this is epoch: " << ep+1 << " error (%) = " << ((-1 / n) * err * 100)  << endl;
		// restart timer
		t.restart();
		disp1.restart(train_size);
 	}
	

 	
 	







}