
// Weight initialization
  		// lecun  : std::sqrt(6.0f / (float) (volume_size));
  		// Xavier : std::sqrt(12.0f / (float) (fan_in + fan_out)); 
#include <random>
double unifRand()
{
    return rand() / double(RAND_MAX);
}
//
// Generate a random number in a real interval.
// param a one end point of the interval
// param b the other end of the interval
// return a inform rand numberin [a,b].
double unifRand(double a, double b)
{
    return (b-a)*unifRand() + a;
}

void initweight(float * W, int Wsize, point_t in, point_t out)
{
		int fan_in =  Wsize * Wsize * in.z;
		int fan_out = out.x * out.y;

		float init_w = sqrt(12.0f / (float) (fan_in+fan_out));  //std::sqrt(2.0f / (float) fan_in);

			for (int idx = 0; idx < fan_in; ++idx)
			{
				W[idx] = unifRand(-init_w, init_w);
			}
}