#pragma once
#include <sstream>
#include <iomanip>
#include <memory>
#include "util.h"
#include "product.h"
#include "activation_function.h"
#include "weight_init.h"

namespace tiny_cnn {
	class layer_base {
	public:
		friend void connection_mismatch(const layer_base&from, const layer_base& to);

		virtual ~layer_base() {}

		layer_base(layer_size_t in_dim, layer_size_t out_dim, size_t weight_dim, size_t bias_dim)
			: parallelize_(true), next_(nullptr), prev_(nullptr),
			weight_init_(std::make_shared<weight_init::xavier>()),
			bias_init_(std::make_shared<weight_init::constant>(0.0)) 
		{
			for (int i = 0; i < CNN_QUEUE_SIZE; ++i) {//initial the state
				outputF_[i] = 0;
				prev_deltaF_[i] = 0;
				current_deltaF_[i] = 0;
			}
			set_size(in_dim, out_dim, weight_dim, bias_dim);
		}

		void connect(std::shared_ptr<layer_base>& tail) {
			if (out_size() != 0 && tail->in_size() != out_size()){
				connection_mismatch(*this, *tail);
			}	
			next_ = tail.get();
			tail->prev_ = this;
		}

		void set_parallelize(bool parallelize) {
			parallelize_ = parallelize;
		}
		//should call this function explicitly after ctor
		void init_weight() {
			weight_init_->fill(&W_, fan_in_size(), fan_out_size());
			bias_init_->fill(&b_, fan_in_size(), fan_out_size());
			clear_diff(CNN_QUEUE_SIZE);
		}
		//getter
		const vec_t& output(int worker_index) const { return output_[worker_index]; }
		const vec_t& delta(int worker_index) const { return prev_delta_[worker_index]; }
		vec_t& weight() { return W_; }
		vec_t& bias() { return b_; }
		vec_t& weight_diff(int index) { return dW_[index]; }
		vec_t& bias_diff(int index) { return db_[index]; }
		bool is_exploded() const { return has_infinite(W_) || has_infinite(b_); }
		layer_base* next() { return next_; }
		layer_base* prev() { return prev_; }


		
		//input dimension
		virtual layer_size_t in_size() const { return in_size_; }
		//output dimension
		virtual  layer_size_t out_size() const { return out_size_; }
		//number of paramerters
		virtual size_t param_size() const { return W_.size() + b_.size(); }
		//number of incoming connections for each output unit
		virtual size_t fan_in_size() const = 0;
		//number of outgoing connections for each input unit
		virtual size_t fan_out_size() const = 0;
		//number of connections
		virtual size_t connection_size() const = 0;
		//input shape(w   h      d)
		virtual index3d<layer_size_t> in_shape() const { return index3d<layer_size_t>(in_size(), 1, 1); }
		//output shape(w  h      d)
		virtual index3d<layer_size_t> out_shape() const { return index3d<layer_size_t>(out_size(), 1, 1); }
		//name of layer. should be unique for each concrete class
		virtual std::string layer_type() const = 0;

		virtual activation::function& activation_function() = 0;

		//setter
		template<typename WeightInit>
		layer_base& weight_init(const WeightInit& f) { weight_init_ = std::make_shared<Weight>(f); return *this; }
		
		template<typename BiasInit>
		layer_base& bias_init(const BiasInit& f) { bias_init_ = std::make_shared<BiasInit>(f); return *this; }

		template<typename WeightInit>
		layer_base& weight_init(std::shared_ptr<WeightInit> f) { weight_init_ = f; return *this; }

		template<typename BiasInit>
		layer_base& bias_init(std::shared_ptr<BiasInit> f) { bias_init_ = f; return *this; }

		//save/load
		virtual void save(std::ostream& os) const {
			if (is_exploded()) throw nn_error("failed to save weights because of infinite weight");
			for (auto w : W_) os << w << " ";
			for (auto b : b_) os << b << " ";
		}

		virtual void load(std::istream& is) {
			for (auto& w : W_) is >> w;
			for (auto& b : b_) is >> b;
		}
		
		//fprop/bprop
		/*
		*return output vector
		*output vector must be stored to a queue output_[worker_index]
		*@param index: where to store 
		*/
		virtual  void forward_propagation(const vec_t & in, size_t worker_index) = 0;

		/*
		*return delta of previous layer(delta = \frac{dE}{da}, a = wx)
		*delta must be stored to prev_delta_[worker_index]
		*/
		virtual const vec_t& back_propagation(const vec_t& current_delta, size_t worker_index) = 0;

		//called after updating weight
		virtual void post_update() {}

		template<typename Optimizer>
		void update_weight(Optimizer* o, int worker_size, size_t batch_size) {
			vec_t tmp;
			if (W_.empty()) return;

			merge(worker_size, batch_size);//to be understand
			o->update(dW_[0], tmp, W_);
			o->update(db_[0], tmp, b_);


			clear_diff(worker_size);
			post_update();
		}
		/******************record the state************************/
		void not_ready_state() {
			for (int i = 0; i < CNN_QUEUE_SIZE; i++) {
				outputF_[i] = 0;  prev_deltaF_[i] = 0;
				current_deltaF_[i] = 0;
			}
		}

		void ready_state() {
			for (int i = 0; i < CNN_QUEUE_SIZE; i++) {
				outputF_[i] = 1;  prev_deltaF_[i] = 1;
				current_deltaF_[i] = 1;
			}
		}

		void update_state() {
			outputF_[0] = 2;
			/*for (int i = 0; i < CNN_QUEUE_SIZE; i++) {
				outputF_[i] = 2;  prev_deltaF_[i] = 2;
				current_deltaF_[i] = 2;
			}*/
		}

		virtual void process() = 0;

	public:
		layer_size_t in_size_;
		layer_size_t out_size_;
		bool parallelize_;

		layer_base* next_;
		layer_base* prev_;
		vec_t a_[CNN_QUEUE_SIZE];// medium result result = w*x +b
		vec_t output_[CNN_QUEUE_SIZE];//last output of current layer,set by fprop
		size_t outputF_[CNN_QUEUE_SIZE];//flag 1:ready  0:not ready
		vec_t prev_delta_[CNN_QUEUE_SIZE];//last delta of previous layer, set by bprop
		size_t prev_deltaF_[CNN_QUEUE_SIZE];////flag 1:ready  0:not ready 2:is updating the weight
		vec_t W_;	//weight vector
		vec_t b_;	//bias vector
		vec_t dW_[CNN_QUEUE_SIZE];
		vec_t db_[CNN_QUEUE_SIZE];
		vec_t current_delta_[CNN_QUEUE_SIZE];
		size_t current_deltaF_[CNN_QUEUE_SIZE];

		std::shared_ptr<weight_init::function> weight_init_; // this is a pointer
		std::shared_ptr<weight_init::function> bias_init_;

	private:
		void set_size(layer_size_t in_dim, layer_size_t out_dim, size_t weight_dim, size_t bias_dim) {
			in_size_ = in_dim;
			out_size_ = out_dim;

			W_.resize(weight_dim);
			b_.resize(bias_dim);

			for (auto& o : output_)		o.resize(out_dim);
			for (auto& a : a_)			a.resize(out_dim);
			for (auto& p : prev_delta_) p.resize(in_dim);
			for (auto& dw : dW_)		dw.resize(weight_dim);
			for (auto& db : db_)		db.resize(bias_dim);
		}

		void merge(size_t worker_size, size_t batch_size) {
			for (size_t i = 1; i < worker_size; i++)
				vectorize::reduce<float_t>(&dW_[i][0], dW_[i].size(), &dW_[0][0]);
			for(size_t i = 1; i < worker_size; i++)
				vectorize::reduce<float_t>(&db_[i][0], db_[i].size(), &db_[0][0]);
			//computer the mean number
			std::transform(dW_[0].begin(), dW_[0].end(), dW_[0].begin(), [&](float_t x) {return x / batch_size; });
			std::transform(db_[0].begin(), db_[0].end(), db_[0].begin(), [&](float_t x) {return x / batch_size; });			
		}

		void clear_diff(size_t worker_size) {
			for (size_t i = 0; i < worker_size; i++) {
				std::fill(dW_[i].begin(), dW_[i].end(), 0.0);
				std::fill(db_[i].begin(), db_[i].end(), 0.0);
			}
		}
	};


	template<typename Activation>
	class layer : public layer_base {
	public:
		layer(layer_size_t in_dim, layer_size_t out_dim, size_t weight_dim, size_t bias_dim)
			: layer_base(in_dim, out_dim, weight_dim, bias_dim) {}

		activation::function& activation_function() override { return h_; }

	protected:
		Activation h_;
	};

	template <typename Char, typename CharTraits>
	std::basic_istream<Char, CharTraits>& operator << (std::basic_istream<Char, CharTraits>& os, layer_base& v) {
		v.save(os);
		return os;
	}

	template<typename Char, typename CharTraits>
	std::basic_istream<Char, CharTraits> & operator >> (std::basic_istream<Char, CharTraits>& os, layer_base& v) {
		v.load(os);
		return os;
	}

	inline void connection_mismatch(const layer_base& from, const layer_base& to) {
		std::ostringstream os;

		os << std::endl;
		os << "output size of Nth layer must be equal to input of (N+1)th layer" << std::endl;
		os << "layerN:   " << std::setw(12) << from.layer_type() << " in:" << from.in_size() << "(" << from.in_shape() << "), " <<
			"out:" << from.out_size() << "(" << from.out_shape() << ")" << std::endl;
		os << "layerN+1: " << std::setw(12) << to.layer_type() << " in:" << to.in_size() << "(" << to.in_shape() << "), " <<
			"out:" << to.out_size() << "(" << to.out_shape() << ")" << std::endl;
		os << from.out_size() << " != " << to.in_size() << std::endl;
		std::string detail_info = os.str();

		throw nn_error("layer dimension mismatch!" + detail_info);
	}

	inline void data_mismatch(const layer_base& layer, const vec_t& data) {
		std::ostringstream os;

		os << std::endl;
		os << "data dimension:    " << data.size() << std::endl;
		os << "network dimension: " << layer.in_size() << "(" << layer.layer_type() << ":" << layer.in_shape() << ")" << std::endl;

		std::string detail_info = os.str();

		throw nn_error("input dimension mismath!" + detail_info);
	}

	inline void pooling_size_mismatch(layer_size_t in_width, layer_size_t in_height, layer_size_t pooling_size) {
		std::ostringstream os;

		os << std::endl;
		os << "WxH:" << in_width << "x" << in_height << std::endl;
		os << "pooling-size:" << pooling_size << std::endl;

		std::string detail_info = os.str();

		throw nn_error("width/height must be multiples of pooling size" + detail_info);
	}

}//namespace tiny_cnn
