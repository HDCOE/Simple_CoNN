
#include "byteswap.h"

uint8_t* write_filet(const char* file_name)
{
	ofstream file(file_name);

	if (file.is_open())
	{
		file << "test \n";
		file << "test2 \n";
	}
	file.close();
}

uint8_t* read_file( const char* szFile )
{
	ifstream file( szFile, ios::binary | ios::ate );
	streamsize size = file.tellg();
	file.seekg( 0, ios::beg );

	if ( size == -1 )
		return nullptr;

	uint8_t* buffer = new uint8_t[size];
	file.read( (char*)buffer, size );
	return buffer;
}
template<int x, int y, int z, int nclass>
struct case_t
{
	tensor_t<float,x, y, z> data;
	tensor_t<float,1,1,nclass> out;
};

vector<case_t<32,32,1,10>> read_dataset(const char * data, const char * label_out)
{
	vector<case_t<32,32,1,10>> cases;


	uint8_t* train_image = read_file( data );
	uint8_t* train_labels = read_file( label_out);

	//uint8_t* train_image = read_file( "train-images.idx3-ubyte" );
	//uint8_t* train_labels = read_file( "train-labels.idx1-ubyte" );

	uint32_t case_count = byteswap_uint32( *(uint32_t*)(train_image + 4) );

	for ( int i = 0; i < (int)case_count; i++ )
	{
		//case_t c {tensor_t<float>( 28, 28, 1 ), tensor_t<float>( 1, 1, 10 )};//case_t c {tensor_t<float>( 28, 28, 1 ), tensor_t<float>( 10, 1, 1 )};
		//case_t c_pad{tensor_t<float>( 32, 32, 1 ), tensor_t<float>( 1, 1, 10 )};
		
		case_t<28,28,1,10> c;
		case_t<32,32,1,10> c_pad;
		uint8_t* img = train_image + 16 + i * (28 * 28);
		uint8_t* label = train_labels + 8 + i;

		for ( int x = 0; x < 28; x++ )
			for ( int y = 0; y < 28; y++ )
				c.data( x, y, 0 ) = img[x + y * 28] / 255.f;

		for ( int b = 0; b < 10; b++ )
			c.out( 0, 0, b ) = *label == b ? 1.0f : 0.0f; //c.out( b, 0, 0 ) = *label == b ? 1.0f : 0.0f;
		// 2 padd
		
		//Padding(c_pad.data,c.data,2);
		Padding(c_pad.data.data(), c_pad.data.size, c.data.data(), 2);
		c_pad.out = c.out;

		cases.push_back( c_pad);
	}
	delete[] train_image;
	delete[] train_labels;

	return cases;
}
/*
void save_layer(vector<tensor_t<float>>& W, ofstream& outfile)
{
	//vector<tensor_t<float>> W;
	//W = la->W;

	 // write to outfile
	  for (int n_filter = 0; n_filter < (int)W.size(); ++n_filter)
	  	 for (int ch = 0; ch < W[n_filter].size.z; ++ch)
	  	 	for (int x = 0; x < W[n_filter].size.x; ++x)
	  	 		for (int y = 0; y < W[n_filter].size.y; ++y)
	  	 			outfile.write ((char *)&W[n_filter](x,y,ch), sizeof(float)); 

	//printf("this is W from orig\n");
  	//print_tensor_vector(W);
}

void save_model(vector<layer*>& layers)
{

	if( remove( "model.dat" ) != 0 )
    	perror( "Error deleting file" );
  	else
  		puts( "File successfully deleted" );

	ofstream outfile ("model.dat",std::ofstream::binary);
	//save_layer(layers[0],outfile);
	//save_layer(layers[3],outfile);

	for (int i = 0; i < (int)layers.size(); ++i)
	{
		switch (layers[i]->type)
		{
			case conv:	save_layer(layers[i]->W, outfile); 	break;
			case fc: 	save_layer(layers[i]->W,outfile);	break;
		}
	}

	outfile.close();
}

void load_layer(vector<tensor_t<float>>& W, ifstream& infile)
{
	//vector<tensor_t<float>> W;
	//W = la->W ;
  //read from file
    for (int n_filter = 0; n_filter < (int)W.size(); ++n_filter)
    	for (int ch = 0; ch < W[n_filter].size.z; ++ch)
    		for (int x = 0; x < W[n_filter].size.x; ++x)
    			for (int y = 0; y < W[n_filter].size.y; ++y)
    				infile.read((char *)&W[n_filter](x,y,ch),sizeof(float));

	//printf("this is W from read\n");
  	//print_tensor_vector(W);   
}

void load_model(vector<layer*>& layers)
{
	ifstream infile("model.dat",std::ofstream::binary);
		//infile.read ((char *)&buffer,sizeof(float));
  // get size of file
  	infile.seekg (0,infile.end);
  	long size = infile.tellg();
  	infile.seekg (0);

  	for (int i = 0; i < (int)layers.size(); ++i)
	{
		switch (layers[i]->type)
		{
			case conv:	load_layer(layers[i]->W, infile); 	break;
			case fc: 	load_layer(layers[i]->W,infile);	break;
		}
	}
  	infile.close();
}
*/