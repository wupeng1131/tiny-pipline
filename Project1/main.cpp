#include <windows.h>
#include<iostream>
#include<boost/timer.hpp>
#include<boost/progress.hpp>
#include"tiny_cnn.h"
//
//int global_count = 0;

using namespace tiny_cnn;
//using namespace std;
using namespace tiny_cnn::activation;
void sample_convet();
DWORD WINAPI p0(LPVOID lpParameter) {
	network<mse, gradient_denscent> *nn = (network<mse, gradient_denscent> *)lpParameter;
	while (1) {
		//((nn->layers_).layers_[0])->f_process();
		((nn->layers_).layers_[1])->f_process();
		((nn->layers_).layers_[2])->f_process();
		((nn->layers_).layers_[3])->f_process();
		/*((nn->layers_).layers_[4])->f_process();
		((nn->layers_).layers_[5])->f_process();
		((nn->layers_).layers_[6])->f_process();
		((nn->layers_).layers_[6])->b_process();
		((nn->layers_).layers_[5])->b_process();
		((nn->layers_).layers_[4])->b_process();
		((nn->layers_).layers_[3])->b_process();
		((nn->layers_).layers_[2])->b_process();
		((nn->layers_).layers_[1])->b_process();
		((nn->layers_).layers_[0])->b_process();*/
		
	}
	//((nn->layers_).layers_[0])->process();
	return 0;
}
DWORD WINAPI p1(LPVOID lpParameter) {
	network<mse, gradient_denscent> *nn = (network<mse, gradient_denscent> *)lpParameter;
	while (1) {
		//((nn->layers_).layers_[0])->f_process();
		//((nn->layers_).layers_[1])->f_process();
		//((nn->layers_).layers_[2])->f_process();
		//((nn->layers_).layers_[3])->f_process();
		((nn->layers_).layers_[4])->f_process();
		((nn->layers_).layers_[5])->f_process();
		((nn->layers_).layers_[6])->f_process();
		nn->b_process();
		((nn->layers_).layers_[6])->b_process();
		((nn->layers_).layers_[5])->b_process();
		((nn->layers_).layers_[4])->b_process();
		((nn->layers_).layers_[3])->b_process();
		((nn->layers_).layers_[2])->b_process();
		((nn->layers_).layers_[1])->b_process();
		//((nn->layers_).layers_[0])->b_process();
		
	}
	//((nn->layers_).layers_[1])->process();
	return 0;
}
DWORD WINAPI p2(LPVOID lpParameter) {
	network<mse, gradient_denscent> *nn = (network<mse, gradient_denscent> *)lpParameter;
	while (1) {
		//((nn->layers_).layers_[0])->f_process();
		//((nn->layers_).layers_[1])->f_process();
		//((nn->layers_).layers_[2])->f_process();
		//((nn->layers_).layers_[3])->f_process();
		((nn->layers_).layers_[4])->f_process();
		((nn->layers_).layers_[5])->f_process();
		((nn->layers_).layers_[6])->f_process();
		nn->b_process();
		((nn->layers_).layers_[6])->b_process();
		((nn->layers_).layers_[5])->b_process();
		//((nn->layers_).layers_[4])->b_process();
		//((nn->layers_).layers_[3])->b_process();
		//((nn->layers_).layers_[2])->b_process();
		//((nn->layers_).layers_[1])->b_process();
		//((nn->layers_).layers_[0])->b_process();
		
	}
	//((nn->layers_).layers_[2])->process();
	return 0;
}
DWORD WINAPI p3(LPVOID lpParameter) {
	network<mse, gradient_denscent> *nn = (network<mse, gradient_denscent> *)lpParameter;
	//((nn->layers_).layers_[3])->process();
	while (1) {
		//((nn->layers_).layers_[0])->f_process();
		//((nn->layers_).layers_[1])->f_process();
		//((nn->layers_).layers_[2])->f_process();
		//((nn->layers_).layers_[3])->f_process();
		//((nn->layers_).layers_[4])->f_process();
		//((nn->layers_).layers_[5])->f_process();
		//((nn->layers_).layers_[6])->f_process();
		//((nn->layers_).layers_[6])->b_process();
		//((nn->layers_).layers_[5])->b_process();
		((nn->layers_).layers_[4])->b_process();
		((nn->layers_).layers_[3])->b_process();
		((nn->layers_).layers_[2])->b_process();
		//((nn->layers_).layers_[1])->b_process();
		//((nn->layers_).layers_[0])->b_process();
		
	}
	return 0;
}
DWORD WINAPI p4(LPVOID lpParameter) {
	network<mse, gradient_denscent> *nn = (network<mse, gradient_denscent> *)lpParameter;
	while (1) {
		//((nn->layers_).layers_[0])->f_process();
		//((nn->layers_).layers_[1])->f_process();
		//((nn->layers_).layers_[2])->f_process();
		//((nn->layers_).layers_[3])->f_process();
		//((nn->layers_).layers_[4])->f_process();
		//((nn->layers_).layers_[5])->f_process();
		//((nn->layers_).layers_[6])->f_process();
		//((nn->layers_).layers_[6])->b_process();
		//((nn->layers_).layers_[5])->b_process();
		//((nn->layers_).layers_[4])->b_process();
		//((nn->layers_).layers_[3])->b_process();
		//((nn->layers_).layers_[2])->b_process();
		((nn->layers_).layers_[1])->b_process();
		//((nn->layers_).layers_[0])->b_process();

	}
	return 0;
}
DWORD WINAPI p5(LPVOID lpParameter) {
	network<mse, gradient_denscent> *nn = (network<mse, gradient_denscent> *)lpParameter;
	((nn->layers_).layers_[5])->process();
	return 0;
}
DWORD WINAPI p6(LPVOID lpParameter) {
	network<mse, gradient_denscent> *nn = (network<mse, gradient_denscent> *)lpParameter;
	((nn->layers_).layers_[6])->process();
	return 0;
}
DWORD WINAPI inspect(LPVOID lpParameter) {
	network<mse, gradient_denscent> *nn = (network<mse, gradient_denscent> *)lpParameter;
	int layer_num = (nn->layers_).layers_.size();
	int count=nn->sample_count;
	while (1) {
		Sleep(10000);
		
		if (count != nn->sample_count) {
			count = nn->sample_count;
			
		}
		else {//choke
			for (int i = 0; i < layer_num; i++) {
				((nn->layers_).layers_[i])->print_state();
			}
			std::cout << std::endl<< ((nn->layers_).layers_[0])->pre_deltaIndex_ <<std::endl;
			std::cout << std::endl;
			for (int i = 0; i < CNN_QUEUE_SIZE; i++) {
				std::cout << ((nn->layers_).layers_[0])->test[i] << "  ";
			}
			std::cout << std::endl;
		}

	}
	return 0;
}


void sample_convet() {
	std::cout << "load models..." << std::endl;
	std::vector<label_t> train_labels, test_labels;
	std::vector<vec_t> train_images, test_images;
	parse_mnist_labels("D:/CODE/TF/tiny-cnn-master/data/train-labels.idx1-ubyte", &train_labels);
	parse_mnist_images("D:/CODE/TF/tiny-cnn-master/data/train-images.idx3-ubyte", &train_images, -1.0, 1.0, 2, 2);
	parse_mnist_labels("D:/CODE/TF/tiny-cnn-master/data/t10k-labels.idx1-ubyte", &test_labels);
	parse_mnist_images("D:/CODE/TF/tiny-cnn-master/data/t10k-images.idx3-ubyte", &test_images, -1.0, 1.0, 2, 2);
	
	std::cout << "start learning" << std::endl;
	boost::progress_display disp(train_images.size());
	
	boost::timer t;

	network<mse, gradient_denscent> nn;
	nn  << convolutional_layer<tan_h>(32, 32, 5, 1, 6)    //1
		<< average_pooling_layer<tan_h>(28, 28, 6, 2)     //2
		<< convolutional_layer<tan_h>(14, 14, 5, 6, 16)   //3
		<< average_pooling_layer<tan_h>(10, 10, 16, 2)    //4
		<< convolutional_layer<tan_h>(5, 5, 5, 16, 120)   //5
		<< fully_connectioned_layer<tan_h>(120, 10);      //6

	/***********load the weight***************/
	//nn.load("LeNet-weights");
	//std::ifstream in_weight;
	//in_weight.open("LeNet-weights");
	//nn.load(in_weight);
	int minibatch_size = CNN_QUEUE_SIZE;

	auto on_enumerate_epoch = [&]() {
		std::cout << t.elapsed() << "s elapsed." << std::endl;
		//tiny_cnn::result res = nn.test(test_images, test_labels);
		//std::cout << nn.optimizer().alpha << "," << res.num_success << "/" << res.num_total << std::endl;

		nn.optimizer().alpha *= 0.85;//decay
		nn.optimizer().alpha = max(0.00001, nn.optimizer().alpha);

		disp.restart(train_images.size());
		t.restart();
	};

	auto on_enumerate_minibatch = [&]() {
		disp += minibatch_size;
		if ((disp.count() % 1000) == 0) {
			//std::cout << disp.count() << "  ";
	//		tiny_cnn::result res = nn.test(test_images, test_labels);
	//		std::cout << nn.optimizer().alpha << "," << res.num_success << "/" << res.num_total << std::endl;
		}
		
		

	};


	HANDLE h0,h1,h2,h3,h4,h5,h6,h7;
	h0 = CreateThread(NULL, 0, p0, &nn, 0, NULL);
	h1 = CreateThread(NULL, 0, p1, &nn, 0, NULL);
	//h2 = CreateThread(NULL, 0, p2, &nn, 0, NULL);
	//h3 = CreateThread(NULL, 0, p3, &nn, 0, NULL);
	//h4 = CreateThread(NULL, 0, p4, &nn, 0, NULL);
	/*h5 = CreateThread(NULL, 0, p5, &nn, 0, NULL);
	h6 = CreateThread(NULL, 0, p6, &nn, 0, NULL);
	*/
	//h7 = CreateThread(NULL, 0, inspect, &nn, 0, NULL);
	



	nn.train(train_images, train_labels, minibatch_size, 1, on_enumerate_minibatch, on_enumerate_epoch);
	//nn.train(train_images, train_labels, minibatch_size, 1);



	std::cout << "end training." << std::endl;

	// test and show results
	nn.test(test_images, test_labels).print_detail(std::cout);

	// save networks
	std::ofstream ofs("LeNet-weights");
	ofs << nn;
}
int main() {
	sample_convet();
}//