#pragma once
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <iterator>
#include <iterator>
#include <map>
#include <set>

//#include "util.h"
#include "loss_function.h"
//#include "activation_function.h"
//#include "product.h"
#include "layer.h"
#include "layers.h"

namespace tiny_cnn {

	struct result {
		result() : num_success(0), num_total(0) {}

		double accuracy() const { 
			return num_success * 100.0 / num_total; 
		}

		template <typename Char,typename CharTraits>
		void print_summary(std::basic_ostream<Char, CharTraits>& os) const {
			os << "accuracy:" << accuracy() << num_success << "/" << num_total << std::endl;
		}
		
		template <typename Char, typename CharTraits>
		void print_detail(std::basic_ostream<Char, CharTraits>& os) {
			print_summary(os);
			auto all_labels = labels();

			os << std::setw(5) << "*" << " ";
			for (auto c : all_labels)
				os << std::setw(5) << c << " ";
			os << std::endl;

			for (auto r : all_labels) {
				os << std::setw(5) << r << " ";
				for (auto c : all_labels)
					os << std::setw(5) << confusion_matrix[r][c] << " ";
				os << std::endl;
			}
		}

		std::set<label_t> labels() const {
			std::set<label_t> all_labels;
			for (auto r : confusion_matrix) {
				all_labels.insert(r.first);
				for (auto c : r.second)
					all_labels.insert(c.first);
			}
			return all_labels;
		}

		int num_success;
		int num_total;
		std::map<label_t, std::map<label_t, int>> confusion_matrix;

	};

	enum grad_check_mode {
		GRAD_CHECK_ALL,
		GRAD_CHECK_RANDOM
	};

	template<typename LossFunction, typename Optimizer>
	class network {
	public:
		typedef LossFunction E;
		explicit network(const std::string &name = "") : name_(name) {};
		//getter
		layer_size_t in_dim() const { return layers_.head()->in_size(); }
		layer_size_t out_dim() const { return layers_.tail()->out_size(); }
		std::string name() const { return name_; }
		Optimizer& optimizer(){ return optimizer_; }

		void init_weight() { layers_.init_weight(); }
		void add(std::shared_ptr<layer_base> layer) { layers_.add(layer); }
		vec_t  predict(const vec_t& in) { return fprop_test(in); }

	    /*
		* training conv-net
		* @param in		array of input data
		* @param t		array of training signals(label or vector)
		* @param epoch  num of training epochs
		* @param on_batch_enumerate callback for each mini-batch enumerate
		* @param on_epoch_enumerate callback for each epoch
		*/

		template <typename OnBatchEnumerate , typename OnEpochEnumerate , typename T>
		bool train(const std::vector<vec_t>& in,
			const std::vector<T>&	 t,
			size_t					 batch_size,
			int						 epoch,
			OnBatchEnumerate         on_batch_enumerate,
			OnEpochEnumerate		 on_epoch_enumerate,
			const bool				 _init_weight = true,
			const int				 nbThreads = CNN_QUEUE_SIZE
		) {
			check_training_data(in, t);
			if (_init_weight) init_weight();
			layers_.set_parallelize(batch_size < CNN_QUEUE_SIZE);
			optimizer_.reset();
			for (int iter = 0; iter < epoch; iter++) {
				for (size_t i = 0; i < in.size(); i += batch_size) {
					train_once(&in[i], &t[i], min(batch_size, in.size() - i), nbThreads);
					on_batch_enumerate();
					//delete the operation  of  is_exploded()
				}
				on_epoch_enumerate();
			}
			return true;
		}

		/*training conv-net without callback*/
		template<typename T>
		bool train(const std::vector<vec_t>& in, const std::vector<T>& t, size_t batch_size = 1, int epoch = 1) {
			return train(in, t, batch_size, epoch, nop, nop);
		}

		/*test and generate confusion-matrix for classification task*/
		result test(const std::vector<vec_t>& in, const std::vector<label_t>& t) {
			result test_result;
			for (size_t i = 0; i < in.size(); i++) {
				const label_t predicted = max_index(predict(in[i]));
				const label_t actual = t[i];

				if (predicted == actual) test_result.num_success++;
				test_result.num_total++;
				test_result.confusion_matrix[predicted][actual]++;
			}
			return test_result;
		}
		/*calculate loss value for regression task*/
		float_t get_loss(const std::vector<vec_t>& in, const std::vector<vec_t>& t) {
			float_t sum_loss = (float_t)0.0;

			for (size_t i = 0; i < in.size(); i++) {
				sum_loss += get_loss(predict(in[i]), t[i]);
			}
			return sum_loss;
		}

		void save(std::ostream& os) const {
			auto l = layers_.head();
			while (l) { l->save(os); l = l->next(); }
		}

		void load(std::istream& is) const {
			auto l = layers_.head();
			while (l) { l->load(is); l = l-> next(); }
		}

		template<typename T>
		const T& at(size_t index) const {
			return layers_.at<T>(index);
		}

		const layer_base* operator[](size_t index) const {
			return layers_[index];
		}

		layer_base* operator[] (size_t index) {
			return layers_[index];
		}

		size_t depth() const {
			return layers_.depth();
		}

		
public:
	void label2vector(const label_t* t, int num, std::vector<vec_t> *vec) const {
		layer_size_t outdim = out_dim();
		assert(num > 0);
		assert(outdim > 0);

		vec->reserve(num);
		for (int i = 0; i < num; i++) {
			assert(t[i] < outdim);
			vec->emplace_back(outdim, target_value_min());
			vec->back()[t[i]] = target_value_max();
		}
	}

	void train_once(const vec_t* in, const label_t* t, int size, const int nbThreads = CNN_QUEUE_SIZE) {
		std::vector<vec_t> v;
		label2vector(t, size, &v);
		train_once(in, &v[0], size, nbThreads);
	}

	float target_value_min() const { return layers_.tail()->activation_function().scale().first; }
	float target_value_max() const { return layers_.tail()->activation_function().scale().second; }
	/* in , t is address   size is num*/
	void train_once(const vec_t* in, const vec_t* t, int size, const int nbThreads = CNN_QUEUE_SIZE) {
	//	if (size == 1) {
	//		bprop(fprop(in[0]), t[0]);
	//		layers_.update_weights(&optimizer_, 1, 1);
	//	}
	//	else {
			train_onebatch(in, t, size, nbThreads);
	//	}
	}



	bool can_process_head(const int& i, const int& j) {
		CNN_UNREFERENCED_PARAMETER(j);
		if (i < CNN_QUEUE_SIZE) 
			return true;
		else 
			return false;
	}
	bool can_process_tail(const int& i, const int& j) {
		CNN_UNREFERENCED_PARAMETER(i);
		if (j >= CNN_QUEUE_SIZE) 
			return false;
		if (layers_.tail()->outputF_[j] == 1)
			return true; 
		else
			return false;
	}
	void train_onebatch(const vec_t* in, const vec_t* t, int batch_size, const int num_tasks = CNN_QUEUE_SIZE) {
		assert(batch_size == CNN_QUEUE_SIZE);
		/*for (int i = 0; i < batch_size; i++) {
			bprop(fprop(in[i], i), t[i], i);// i is index
		}*/
		int i = 0; int j = 0;// i control forward propagation     j control backward propagation
		while (j < batch_size) {//tail has complete
			if (can_process_head(i,j)) {
				//process head
				fprop(in[i], i);
					i++;
					//if (i == CNN_QUEUE_SIZE - 1) { std::cout << "input finished"; }
			}
			if (can_process_tail(i,j)) {
				//process head
				bprop(layers_.tail()->output_[j], t[j], j);
				j++;
			}
		}

		while (layers_.head()->current_deltaF_[CNN_QUEUE_SIZE - 1] != 1) {}
		
		/*for (int i = 0; i < batch_size; i++) {
			bprop(fprop(in[i]), t[i])
		}
		layers_.update_weights(&optimizer_, num_tasks, batch_size);*/
		/*task_group g;                  //what is task group??

		int data_per_thread = batch_size / num_tasks;  //number of data to use in each thread
		int remaining = batch_size;

		// divide batch data and invoke [num_tasks] tasks
		for (int i = 0; i < num_tasks; i++) {
			int num = i == num_tasks - 1 ? remaining : data_per_thread;
			//if last task:remaining      else:average num
			g.run([=] {                //copy
				//for (int j = 0; j < num; j++) bprop( (in[j], i), t[j], i);
				int j = 0;
				while (j < num) {
					process(i, j, num);//i is a task, num is number per task, j is  data per task
				}
			});

			remaining -= num;
			in += num;
			t += num;
		}

		assert(remaining == 0);//???
		g.wait();
		*/
		//before update the weight it should set a flag that not allow paragation
		//for (auto pl : layers_.layers_) {
		//	pl->update_state();
		//}
		// merge all dW and update W by optimizer
		layers_.update_weights(&optimizer_, num_tasks, batch_size);
		// restart state
		for (auto pl : layers_.layers_) {
			pl->update_state();
		}
	}

	const vec_t& fprop_test(const vec_t& in, int index = 0) {
		if (in.size() != (size_t)in_dim())  data_mismatch(*layers_[0], in);
		auto l = layers_.head();
		l->forward_propagation(in, index);
		l = l->next();
		for (int i = 1; i < layers_.layers_.size(); i++)
			//vec_t m_out = l->prev()->output(0);
		{
			l->forward_propagation(l->prev()->output(index), index);
			l = l->next();
		}

		return layers_.tail()->output_[index];
	}
	const vec_t& fprop(const vec_t& in,int index = 0) {
		if (in.size() != (size_t)in_dim() )  data_mismatch(*layers_[0], in);
		layers_.head()->output_[index] = in;
		layers_.head()->outputF_[index] = 1;
		return in;
		//return layers_.head()->forward_propagation(in, index);
	}

	template<typename Activation>
	bool is_canonical_link(const Activation& h) {
		if (typeid(h) == typeid(activation::sigmoid) && typeid(E) == typeid(cross_entropy)) return true;
		if (typeid(h) == typeid(activation::tan_h) && typeid(E) == typeid(cross_entropy)) return true;
		if (typeid(h) == typeid(activation::identity) && typeid(E) == typeid(mse)) return true;
		if (typeid(h) == typeid(activation::softmax) && typeid(E) == typeid(cross_entropy_multiclass)) return true;
		return false;
	}

	void bprop(const vec_t& out, const vec_t& t, int idx = 0) {
		vec_t delta(out_dim());
		const activation::function& h = layers_.tail()->activation_function();

		if (is_canonical_link(h)) {
			for_i(out_dim(), [&](int i) {delta[i] = out[i] - t[i]; });
		}
		else {
			vec_t dE_dy = gradient<E>(out, t);	//delta by loss function
			// delta = dE/da = (dE/dy) * (dy/da)
			for (size_t i = 0; i < out_dim(); i++) { //for all output
				vec_t dy_da = h.df(out, i);// de/dy is a vector, and dy/da is a vector
				delta[i] = vectorize::dot(&dE_dy[0], &dy_da[0], out_dim());
		}
		}
		//layers_.tail()->back_propagation(delta, idx);
		layers_.tail()->current_delta_[idx] = delta;
		layers_.tail()->current_deltaF_[idx] = 1;
	}

	float_t get_loss(const vec_t& out, const vec_t& t) {
			float_t e = 0.0;
			assert(out.size() == t.size());
			for_i(out.size(), [&](int i) {e += E::f(out[i], t[i]); })
			return e;
		}

	template <typename T>
	void check_training_data(const std::vector<vec_t>& in, const std::vector<T>& t) {
		layer_size_t dim_in = in_dim();
		layer_size_t dim_out = out_dim();

		if (in.size() != t.size())
			throw nn_error("number of training data must be equal to label data");

		size_t num = in.size();

		for (size_t i = 0; i < num; i++) {
			if (in[i].size() != dim_in)
				throw nn_error(format_str("input dimension mismatch!\n dim(data[%u])=%d, dim(network input)=%u", i, in[i].size(), dim_in));

			check_t(i, t[i], dim_out);
		}
	}
		void check_t(size_t i, const vec_t& t, layer_size_t dim_out) {
			if (t.size() != dim_out) {
				throw nn_error(format_str("output dimension mismatch!\n dim(target[%u])=%u, dim(network output) size=%u", i, t.size(), dim_out));
			}
		}

		void check_t(size_t i, label_t t, layer_size_t dim_out) {
			if (t >= dim_out) {
				std::ostringstream os;
				os << format_str("t[%u]=%u, dim(network output)=%u", i, t, dim_out) << std::endl;
				os << "in classification task, dim(network output) must be greater than max class id." << std::endl;
				if (dim_out == 1)
					os << std::endl << "(for regression, use vector<vec_t> instead of vector<label_t> for training signal)" << std::endl;

				throw nn_error("output dimension mismatch!\n " + os.str());
			}
		}

		std::string name_;
		Optimizer optimizer_;
		layers layers_;





		inline void data_mismatch(const layer_base& layer, const vec_t& data) {
			std::ostringstream os;
			os << std::endl;
			os << "data dimension:   " << data.size() << std::endl;
			os << "network dimension: " << layer.in_size() << "(" << layer.layer_type() << ":" << layer.in_shape() << ")" << std::endl;
			std::string detail_info = os.str();
			throw nn_error("input dimension mismath!" + detail_info);
		}
	};

	template <typename L, typename O, typename Layer>
	network<L, O>& operator <<(network<L, O>& n, const Layer&& l) {
		n.add(std::make_shared<Layer>(l));
		return n;
	}

	template<typename L, typename O, typename Layer>
	network<L, O> operator << (network<L, O>& n, Layer& l) {
		n.add(std::make_shared<Layer>(l));
		return n;
	}

	template <typename L, typename O, typename Char, typename CharTraits>
	std::basic_ostream<Char, CharTraits>& operator << (std::basic_ostream<Char, CharTraits>& os, const network<L, O>& n) {
		n.save(os);
		return os;
	}

	template <typename L, typename O, typename Char, typename CharTraits>
	std::basic_istream<Char, CharTraits>& operator >> (std::basic_istream<Char, CharTraits>& os, network<L, O>& n) {
		n.load(os);
		return os;
	}




}
