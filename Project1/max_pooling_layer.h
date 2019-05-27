#pragma once

#include "util.h"
#include "partial_connected_layer.h"
#include "activation_function.h"

namespace tiny_cnn {
	template<typename Activation>
	class max_pooling_layer :public layer<Activation> {
	public:
		typedef layer<Activation> Base;
		max_pooling_layer(layer_size_t in_width, layer_size_t in_height, layer_size_t in_channels, layer_size_t pooling_size)
			: Base(in_width * in_height * in_channels,
				in_width * in_height * in_channels / sqr(pooling_size),
				0, 0),
			in_(in_width, in_height, in_channels),
			out_(in_width / pooling_size, in_height / pooling_size, in_channels),
			pool_size_(pooling_size)
		{
			if ((in_width % pooling_size) || (in_height % pooling_size))
				pooling_size_mismatch(in_width, in_height, pooling_size);

			init_connection(pooling_size);
		}

		size_t fan_in_size() const override {
			return out2in_[0].size();
		}

		size_t fan_out_size() const override {
			return 1;
		}

		size_t connection_size() const override {
			return out2in_[0].size() * out2in_.size();
		}

		virtual void forward_propagation(const vec_t& in, size_t index) {
			for_(parallelize_, 0, out_size_, [&](const blocked_range& r) {
				for (int i = r.begin(); i < r.end(); i++) {
					const auto& in_index = out2in_[i];
					float_t max_value = std::numeric_limits<float_t>::lowest();

					for (auto j : in_index) {
						if (in[j] > max_value) {
							max_value = in[j];
							out2inmax_[i] = j;
						}
					}
					output_[index][i] = max_value;
				}
			});
			//return next_ ? next_->forward_propagation(output_[index], index) : output_[index];
			//return output_[index];
		}

		virtual const vec_t& back_propagation(const vec_t& current_delta, size_t index) {
			const vec_t& prev_out = prev_->output(index);
			const activation::function& prev_h = prev_->activation_function();
			vec_t& prev_delta = prev_delta_[index];

			for_(parallelize_, 0, in_size_, [&](const blocked_range& r) {
				for (int i = r.begin(); i != r.end(); i++) {
					int outi = in2out_[i];
					prev_delta[i] = (out2inmax_[outi] == i) ? current_delta[outi] * prev_h.df(prev_out[i]) : 0.0;
				}
			});
			//return prev_->back_propagation(prev_delta_[index], index);
			return prev_delta_[index];
		}

		/******************************************/
		void initIndex(int& outputIndex, int& pre_deltaIndex) {
			if (outputF_[0] == 2) { outputIndex = 0;
			pre_deltaIndex = 0;
			not_ready_state();
			}//updateing state, reset the index
		}
		bool can_forward(const int& outputIndex, const int& pre_deltaIndex) {
			CNN_UNREFERENCED_PARAMETER(pre_deltaIndex);
			if (outputIndex >= CNN_QUEUE_SIZE) return false;
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

		void f_process()override {
			initIndex(outputIndex_, pre_deltaIndex_);
			if (can_forward(outputIndex_, pre_deltaIndex_)) {
			#ifdef __PRINT_TIME
				m_time t;
			#endif // __PRINT_TIME

			forward_propagation(prev_->output_[outputIndex_], outputIndex_);

			#ifdef __PRINT_TIME
				if (f_print_) {
					double tmp = t.elapsed();
					f_time += tmp;
					f_print_--;
					if (f_print_ == 0) std::cout << "f_layer" << layerIndex_ << ":" << f_time / PRINT_COUNT << "ms" << std::endl;
				}
			#endif // __PRINT_TIME

			outputF_[outputIndex_] = 1;
			outputIndex_++;
			}
		}
		void b_process()override {
			initIndex(outputIndex_, pre_deltaIndex_);
			if (can_backward(outputIndex_, pre_deltaIndex_)) {

			#ifdef __PRINT_TIME
				m_time t;
			#endif // __PRINT_TIME

			back_propagation(next_->prev_delta_[pre_deltaIndex_], pre_deltaIndex_);

			#ifdef __PRINT_TIME
				if (b_print_) {
					double tmp = t.elapsed();
					b_time += tmp;
					b_print_--;
					if (b_print_ == 0) std::cout << "b_layer" << layerIndex_ << ":" << b_time / PRINT_COUNT << "ms" << std::endl;
				}
			#endif // __PRINT_TIME
				

			prev_deltaF_[pre_deltaIndex_] = 1;
			current_deltaF_[pre_deltaIndex_] = 1;
			pre_deltaIndex_++;
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
					if (outputIndex == CNN_QUEUE_SIZE - 1) { std::cout << "maxpooling forward finished" << std::endl; }
					outputIndex++;
				}

				if (can_backward(outputIndex, pre_deltaIndex)) {
					//backward process
					back_propagation(next_->prev_delta_[pre_deltaIndex], pre_deltaIndex);
					prev_deltaF_[pre_deltaIndex] = 1;
					pre_deltaIndex++;
				}
			}
		}


		index3d<layer_size_t> in_shape() const override { return in_; }
		index3d<layer_size_t> out_shape() const override { return out_; }
		std::string layer_type() const override { return "max-pool"; }
		size_t pool_size() const { return pool_size_; }

	private:
		size_t pool_size_;
		std::vector<std::vector<int> > out2in_; // mapping out => in (1:N)
		std::vector<int> in2out_; // mapping in => out (N:1)
		std::vector<int> out2inmax_; // mapping out => max_index(in) (1:1)
		index3d<layer_size_t> in_;
		index3d<layer_size_t> out_;

		void connect_kernel(layer_size_t pooling_size, layer_size_t outx, layer_size_t outy, layer_size_t  c)
		{
			for (layer_size_t dy = 0; dy < pooling_size; dy++) {
				for (layer_size_t dx = 0; dx < pooling_size; dx++) {
					layer_size_t in_index = in_.get_index(outx * pooling_size + dx, outy * pooling_size + dy, c);
					layer_size_t out_index = out_.get_index(outx, outy, c);
					in2out_[in_index] = out_index;
					out2in_[out_index].push_back(in_index);
				}
			}
		}

		void init_connection(layer_size_t pooling_size)
		{
			in2out_.resize(in_.size());
			out2in_.resize(out_.size());
			out2inmax_.resize(out_.size());
			for (layer_size_t c = 0; c < in_.depth_; ++c)
				for (layer_size_t y = 0; y < out_.height_; ++y)
					for (layer_size_t x = 0; x < out_.width_; ++x)
						connect_kernel(pooling_size, x, y, c);
		}
	};
}
