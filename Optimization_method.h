

#define learning_rate  0.01 * (1.0 / mini_batch)  //0.000167 // 0.000625 // 0.01 * (1.0/16) learning_rate * (1 / batch size)
#define momentum 0.9
#define beta1 0.9
#define beta2 0.999

static void SGD_update(float& W, float& gradient , float& V_m)
{
	W = W - learning_rate * gradient;
	gradient = 0;
}
static void SGD_momentum(float& W, float& gradient , float& V_m)
{
	V_m = momentum * V_m - learning_rate * gradient;

	W += V_m;
	
	gradient = 0;
}

static void RMSprop(float& W, float& gradient , float& V_m)
{
	V_m = momentum * V_m + (1 - momentum)*pow(gradient,2);

	W = W - learning_rate * (gradient / (sqrt(V_m) + 0.0001));

	gradient = 0;
}

static void Adam(float& W, float& gradient , float& V_m, float& S_m)
{
	V_m = beta1 * V_m + (1-beta1) * gradient;
	S_m = beta2 * S_m + (1-beta2) * pow(gradient,2);

	//corrected
	float v_c = V_m / (1-beta1);
	float s_c = S_m / (1-beta2);

	// update
	W = W - learning_rate * (v_c / (sqrt(s_c) + 0.000001));

	gradient = 0;
}