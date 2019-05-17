#pragma once
#include "layer.h"
#include "product.h"
#include "dropout.h"

namespace tiny_cnn {

	template<typename Activation, typename Filter = filter_none>
	class fully_connectioned_layer :public layer<Activation> {
	public:
		typedef layer<Activation> Base;
		fully_connectioned_layer(layer_size_t in_dim, layer_size_t out_dim)
			: Base(in_dim, out_dim, size_t(in_dim) * out_dim, out_dim), filter_(out_dim) {}

		size_t connection_size() const override {
			return size_t(in_size_) * out_size_ + out_size_;
		}

		size_t fan_in_size() const override {
			return in_size_;
		}

		size_t fan_out_size() const override {
			return out_size_;
		}

		void forward_propagation(const vec_t& in, size_t index) {
			vec_t &a = a_[index];
			vec_t &out = output_[index];

			for_i(parallelize_, out_size_, [&](int i){
				a[i] = 0.0;
				for (int c = 0; c < in_size_; c++)
					a[i] += W_[c*out_size_ + i] * in[c];
				a[i] += b_[i];
			});

			for_i(parallelize_, out_size_, [&](int i) {
				out[i] = h_.f(a, i);
			});
			auto& this_out = filter_.filter_fprop(out, index);
			//return this_out;
			//return next_ ? next_->forward_propagation(this_out, index) : this_out;
		}

		const vec_t& back_propagation(const vec_t& current_delta, size_t index) {
			const vec_t& curr_delta = filter_.filter_bprop(current_delta, index);
			const vec_t&  prev_out = prev_->output(index);
			const activation::function& prev_h = prev_->activation_function();
			vec_t& prev_delta = prev_delta_[index];
			vec_t& dW = dW_[index];
			vec_t& db = db_[index];

			for (int c = 0; c < this->fan_in_size(); c++) {
				prev_delta[c] = vectorize::dot(&curr_delta[0], &W_[c*out_size_], out_size_);
				prev_delta[c] *= prev_h.df(prev_out[c]);
			}

			for_(parallelize_, 0, out_size_, [&](const blocked_range& r) {
				for (int c = 0; c < in_size_; c++)
					vectorize::muladd(&curr_delta[0], prev_out[c], r.end() - r.begin(), &dW[c*out_size_ + r.begin()]);
			
				for (int i = r.begin(); i < r.end(); i++)
					db[i] += curr_delta[i];
			});

			//return prev_->back_propagation(prev_delta_[index], index);
			return prev_delta_[index];
		}

		std::string layer_type() const override { return "fully-connected"; }
		/******************************************/
		void initIndex(int& outputIndex, int& pre_deltaIndex) {
			if (outputF_[0] == 2) { outputIndex = 0; pre_deltaIndex = 0;
			not_ready_state();
			}//updateing state, reset the index
		}
		bool can_forward(const int& outputIndex, const int& pre_deltaIndex) {
			CNN_UNREFERENCED_PARAMETER(pre_deltaIndex);
			if (outputIndex >= CNN_QUEUE_SIZE) return false;//bug: index overflow
			if (prev_->outputF_[outputIndex] == 1) return true;
			else return false;//not compute and not full
		}
		bool can_backward(const int& outputIndex, const int& pre_deltaIndex) {
			if (pre_deltaIndex >= CNN_QUEUE_SIZE) return false;
			if (next_) {
				if (next_->prev_deltaF_[pre_deltaIndex] == 1)
					return true;
				else
					return false;
			}
			else {
				if (current_deltaF_[pre_deltaIndex] == 1)
					return true;
				else
					return false;
			}
			
		}

		void process() {
			int outputIndex = 0; int pre_deltaIndex = 0;
			while (1) {
				initIndex(outputIndex, pre_deltaIndex);
				if (can_forward(outputIndex, pre_deltaIndex)) {
					//forward process
					forward_propagation(prev_->output_[outputIndex], outputIndex);
					outputF_[outputIndex] = 1;
					//if (outputIndex == CNN_QUEUE_SIZE - 1) { std::cout << "fully_connected forward finished" << std::endl; }
					outputIndex++;
				}

				if (next_) {
					if (can_backward(outputIndex, pre_deltaIndex)) {
						//backward process
						back_propagation(next_->prev_delta_[pre_deltaIndex], pre_deltaIndex);
						//if (pre_deltaIndex == CNN_QUEUE_SIZE - 1) { std::cout << "fully_connected backward finished" << std::endl; }
						prev_deltaF_[pre_deltaIndex] = 1;
						
						pre_deltaIndex++;
					}
				}
				else {//the last fully connected
					if (can_backward(outputIndex, pre_deltaIndex)) {
						back_propagation(current_delta_[pre_deltaIndex], pre_deltaIndex);
						//if (pre_deltaIndex == CNN_QUEUE_SIZE - 1) { std::cout << "last_fully_connected backward finished" << std::endl; }
						prev_deltaF_[pre_deltaIndex] = 1;

						pre_deltaIndex++;
					}
				}
			}
		}

	protected:
		Filter filter_;

	};
			

}