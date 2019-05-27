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
static void usage(const char * argv0) {
	std::cout << "usage:" << argv0 << " --data_path path_to_dataset_floder"
		<< " --learning_rate 0.01"
		<< "--epochs 30"
		<< " --minibatch_size 10"
		<< " -- backend_type internal" << std::endl;
}
DWORD WINAPI p0(LPVOID lpParameter) {
	network<mse, gradient_denscent> *nn = (network<mse, gradient_denscent> *)lpParameter;
	while (1) {
		((nn->layers_).layers_[1])->f_process();
		((nn->layers_).layers_[2])->f_process();
		((nn->layers_).layers_[3])->f_process();
		((nn->layers_).layers_[4])->f_process();
		((nn->layers_).layers_[5])->f_process();
		((nn->layers_).layers_[6])->f_process();
		((nn->layers_).layers_[7])->f_process();
		((nn->layers_).layers_[8])->f_process();
		nn->b_process();
		((nn->layers_).layers_[8])->b_process();
		((nn->layers_).layers_[7])->b_process();
		((nn->layers_).layers_[6])->b_process();
		((nn->layers_).layers_[5])->b_process();
		((nn->layers_).layers_[4])->b_process();
		//((nn->layers_).layers_[3])->b_process();
		//((nn->layers_).layers_[2])->b_process();
		//((nn->layers_).layers_[1])->b_process();


	}
	//((nn->layers_).layers_[0])->process();
	return 0;
}
DWORD WINAPI p1(LPVOID lpParameter) {
	network<mse, gradient_denscent> *nn = (network<mse, gradient_denscent> *)lpParameter;
	while (1) {
		/*((nn->layers_).layers_[0])->f_process();
		((nn->layers_).layers_[1])->f_process();
		((nn->layers_).layers_[2])->f_process();
		((nn->layers_).layers_[3])->f_process();
		((nn->layers_).layers_[4])->f_process();
		((nn->layers_).layers_[5])->f_process();
		((nn->layers_).layers_[6])->f_process();
		((nn->layers_).layers_[7])->f_process();
		((nn->layers_).layers_[8])->f_process();
		nn->b_process();
		((nn->layers_).layers_[8])->b_process();
		((nn->layers_).layers_[7])->b_process();
		((nn->layers_).layers_[6])->b_process();
		((nn->layers_).layers_[5])->b_process();
		((nn->layers_).layers_[4])->b_process();*/
		((nn->layers_).layers_[3])->b_process();
		/*((nn->layers_).layers_[2])->b_process();
		((nn->layers_).layers_[1])->b_process();
		((nn->layers_).layers_[0])->b_process();*/

	}
	//((nn->layers_).layers_[1])->process();
	return 0;
}
DWORD WINAPI p2(LPVOID lpParameter) {
	network<mse, gradient_denscent> *nn = (network<mse, gradient_denscent> *)lpParameter;
	while (1) {
		
		((nn->layers_).layers_[2])->b_process();
		((nn->layers_).layers_[1])->b_process();
	

	}
	//((nn->layers_).layers_[2])->process();
	return 0;
}
DWORD WINAPI p3(LPVOID lpParameter) {
	network<mse, gradient_denscent> *nn = (network<mse, gradient_denscent> *)lpParameter;
	//((nn->layers_).layers_[3])->process();
	while (1) {
		((nn->layers_).layers_[0])->f_process();
		((nn->layers_).layers_[1])->f_process();
		((nn->layers_).layers_[2])->f_process();
		((nn->layers_).layers_[3])->f_process();
		((nn->layers_).layers_[4])->f_process();
		((nn->layers_).layers_[5])->f_process();
		((nn->layers_).layers_[6])->f_process();
		((nn->layers_).layers_[7])->f_process();
		((nn->layers_).layers_[8])->f_process();
		nn->b_process();
		((nn->layers_).layers_[8])->b_process();
		((nn->layers_).layers_[7])->b_process();
		((nn->layers_).layers_[6])->b_process();
		((nn->layers_).layers_[5])->b_process();
		((nn->layers_).layers_[4])->b_process();
		((nn->layers_).layers_[3])->b_process();
		((nn->layers_).layers_[2])->b_process();
		((nn->layers_).layers_[1])->b_process();
		((nn->layers_).layers_[0])->b_process();

	}
	return 0;
}
DWORD WINAPI p4(LPVOID lpParameter) {
	network<mse, gradient_denscent> *nn = (network<mse, gradient_denscent> *)lpParameter;
	while (1) {
		((nn->layers_).layers_[0])->f_process();
		((nn->layers_).layers_[1])->f_process();
		((nn->layers_).layers_[2])->f_process();
		((nn->layers_).layers_[3])->f_process();
		((nn->layers_).layers_[4])->f_process();
		((nn->layers_).layers_[5])->f_process();
		((nn->layers_).layers_[6])->f_process();
		((nn->layers_).layers_[7])->f_process();
		((nn->layers_).layers_[8])->f_process();
		nn->b_process();
		((nn->layers_).layers_[8])->b_process();
		((nn->layers_).layers_[7])->b_process();
		((nn->layers_).layers_[6])->b_process();
		((nn->layers_).layers_[5])->b_process();
		((nn->layers_).layers_[4])->b_process();
		((nn->layers_).layers_[3])->b_process();
		((nn->layers_).layers_[2])->b_process();
		((nn->layers_).layers_[1])->b_process();
		((nn->layers_).layers_[0])->b_process();

	}
	return 0;
}
DWORD WINAPI p5(LPVOID lpParameter) {
	network<mse, gradient_denscent> *nn = (network<mse, gradient_denscent> *)lpParameter;
	while (1) {
		((nn->layers_).layers_[0])->f_process();
		((nn->layers_).layers_[1])->f_process();
		((nn->layers_).layers_[2])->f_process();
		((nn->layers_).layers_[3])->f_process();
		((nn->layers_).layers_[4])->f_process();
		((nn->layers_).layers_[5])->f_process();
		((nn->layers_).layers_[6])->f_process();
		((nn->layers_).layers_[7])->f_process();
		((nn->layers_).layers_[8])->f_process();
		nn->b_process();
		((nn->layers_).layers_[8])->b_process();
		((nn->layers_).layers_[7])->b_process();
		((nn->layers_).layers_[6])->b_process();
		((nn->layers_).layers_[5])->b_process();
		((nn->layers_).layers_[4])->b_process();
		((nn->layers_).layers_[3])->b_process();
		((nn->layers_).layers_[2])->b_process();
		((nn->layers_).layers_[1])->b_process();
		((nn->layers_).layers_[0])->b_process();

	}
	return 0;
}
DWORD WINAPI p6(LPVOID lpParameter) {
	network<mse, gradient_denscent> *nn = (network<mse, gradient_denscent> *)lpParameter;
	while (1) {
		((nn->layers_).layers_[0])->f_process();
		((nn->layers_).layers_[1])->f_process();
		((nn->layers_).layers_[2])->f_process();
		((nn->layers_).layers_[3])->f_process();
		((nn->layers_).layers_[4])->f_process();
		((nn->layers_).layers_[5])->f_process();
		((nn->layers_).layers_[6])->f_process();
		((nn->layers_).layers_[7])->f_process();
		((nn->layers_).layers_[8])->f_process();
		nn->b_process();
		((nn->layers_).layers_[8])->b_process();
		((nn->layers_).layers_[7])->b_process();
		((nn->layers_).layers_[6])->b_process();
		((nn->layers_).layers_[5])->b_process();
		((nn->layers_).layers_[4])->b_process();
		((nn->layers_).layers_[3])->b_process();
		((nn->layers_).layers_[2])->b_process();
		((nn->layers_).layers_[1])->b_process();
		((nn->layers_).layers_[0])->b_process();

	}
	return 0;
}
DWORD WINAPI inspect(LPVOID lpParameter) {
	network<mse, gradient_denscent> *nn = (network<mse, gradient_denscent> *)lpParameter;
	int layer_num = (nn->layers_).layers_.size();
	int count = nn->sample_count;
	while (1) {
		Sleep(10000);

		if (count != nn->sample_count) {
			count = nn->sample_count;

		}
		else {//choke
			for (int i = 0; i < layer_num; i++) {
				((nn->layers_).layers_[i])->print_state();
			}
			std::cout << std::endl << ((nn->layers_).layers_[0])->pre_deltaIndex_ << std::endl;
			std::cout << std::endl;
			for (int i = 0; i < CNN_QUEUE_SIZE; i++) {
				std::cout << ((nn->layers_).layers_[0])->test[i] << "  ";
			}
			std::cout << std::endl;
		}

	}
	return 0;
}
void construct_net(network<mse, gradient_denscent>  &nn) {
	

	const size_t n_fmaps = 32;
	const size_t n_fmaps2 = 64;
	const size_t n_fc = 64;
	//8 layers
	nn << convolutional_layer<relu>(32, 32, 5,5, 3, n_fmaps, padding::same, true, 1, 1);
	nn << max_pooling_layer<identity>(32, 32, n_fmaps, 2);
	nn << convolutional_layer<relu>(16, 16, 5,5, n_fmaps, n_fmaps, padding::same, true, 1, 1);
	nn << max_pooling_layer<identity>(16, 16, n_fmaps, 2);
	nn << convolutional_layer<relu>(8, 8, 5,5, n_fmaps, n_fmaps2,padding::same, true, 1, 1);
	nn << max_pooling_layer<identity>(8, 8, n_fmaps2, 2);
	nn << fully_connectioned_layer<relu>(4 * 4 * n_fmaps2, n_fc);
	nn << fully_connectioned_layer<softmax>(n_fc, 10);
}


void train_cifar10(std::string data_dir_path,
	double learning_rate,
	const int n_train_epochs,
	const int n_minibatch,
	std::ostream &log) {
	//specify loss-function and learning strategy

	network<mse, gradient_denscent> nn;
	construct_net(nn);
	std::cout << "load models..." << std::endl;

	std::vector<label_t> train_labels, test_labels;
	std::vector<vec_t> train_images, test_images;

	for (int i = 1; i <= 5; i++) {
		parse_cifar10(data_dir_path + "/data_batch_" + std::to_string(i) + ".bin",
			&train_images, &train_labels, -1.0, 1.0, 0, 0);
	}

	parse_cifar10(data_dir_path + "/test_batch.bin", &test_images, &test_labels,
		-1.0, 1.0, 0, 0);

	std::cout << "start learning" << std::endl;

	boost::progress_display disp(train_images.size());
	boost::timer t;

	//optimizer<false>.alpha *=
	//	static_cast<tiny_dnn::float_t>(sqrt(n_minibatch) * learning_rate);
	int epoch = 1;
	// create callback
	auto on_enumerate_epoch = [&]() {
		std::cout << "Epoch " << epoch << "/" << n_train_epochs << " finished. "
			<< t.elapsed() << "s elapsed." << std::endl;
		++epoch;
		result res = nn.test(test_images, test_labels);
		log << res.num_success << "/" << res.num_total << std::endl;

		disp.restart(train_images.size());
		t.restart();
	};

	auto on_enumerate_minibatch = [&]() { disp += n_minibatch; };

	HANDLE h0, h1, h2, h3, h4, h5, h6, h7;
	h0 = CreateThread(NULL, 0, p0, &nn, 0, NULL);
	h1 = CreateThread(NULL, 0, p1, &nn, 0, NULL);
	h2 = CreateThread(NULL, 0, p2, &nn, 0, NULL);
	//h3 = CreateThread(NULL, 0, p3, &nn, 0, NULL);
	//h4 = CreateThread(NULL, 0, p4, &nn, 0, NULL);
	/*h5 = CreateThread(NULL, 0, p5, &nn, 0, NULL);
	h6 = CreateThread(NULL, 0, p6, &nn, 0, NULL);
	*/
	//h7 = CreateThread(NULL, 0, inspect, &nn, 0, NULL);

	// training
	nn.train(train_images, train_labels,
		 n_minibatch,n_train_epochs,
		on_enumerate_minibatch, on_enumerate_epoch);

	std::cout << "end training." << std::endl;

	// test and show results
	nn.test(test_images, test_labels).print_detail(std::cout);
	// save networks
	std::ofstream ofs("cifar-weights");
	ofs << nn;
}




int main(int argc, char **argv) {
	//sample_convet();
	double learning_rate = 0.01;
	int epochs = 3;
	std::string data_path = "";
	int minibatch_size = CNN_QUEUE_SIZE;
	

	if (argc == 2) {
		std::string argname(argv[1]);
		if (argname == "--help" || argname == "-h") {
			usage(argv[0]);
			return 0;
		}
	}

	for (int count = 1; count + 1 < argc; count += 2) {
		std::string argname(argv[count]);
		if (argname == "--learning_rate") {
			learning_rate = atof(argv[count + 1]);
		}
		else if (argname == "--epochs") {
			//epochs = atoi(argv[count + 1]);
			continue;
		}
		else if (argname == "--minibatch_size") {
			//minibatch_size = atoi(argv[count + 1]);
			continue;
		}
		else if (argname == "--data_path") {
			data_path = std::string(argv[count + 1]);
		}
		else {
			std::cerr << "Invalid parameter specified -\"" << argname << "\"" << std::endl;
			usage(argv[0]);
			return -1;
		}
	}

	if (data_path == "") {
		std::cerr << "Data path not specified." << std::endl;
		usage(argv[0]);
		return -1;
	}

	if (learning_rate <= 0) {
		std::cerr
			<< "Invalid learning rate. The learning rate must be greater than 0."
			<< std::endl;
		return -1;
	}
	if (epochs <= 0) {
		std::cerr << "Invalid number of epochs. The number of epochs must be "
			"greater than 0."
			<< std::endl;
		return -1;
	}
	if (minibatch_size <= 0 || minibatch_size > 50000) {
		std::cerr
			<< "Invalid minibatch size. The minibatch size must be greater than 0"
			" and less than dataset size (50000)."
			<< std::endl;
		return -1;
	}
	std::cout << "Running with the following parameters:" << std::endl
		<< "Data path: " << data_path << std::endl
		<< "Learning rate: " << learning_rate << std::endl
		<< "Minibatch size: " << minibatch_size << std::endl
		<< "Number of epochs: " << epochs << std::endl
		<< std::endl;

	//try {
	train_cifar10(data_path, learning_rate, epochs, minibatch_size , std::cout);
	//}
	//catch (tiny_dnn::nn_error &err) {
	//	std::cerr << "Exception:" << err.what() << std::endl;
	//	}


}//
