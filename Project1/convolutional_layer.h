#pragma once
#include "util.h"
#include "partial_connected_layer.h"

namespace tiny_cnn {
	struct connection_table {
		connection_table() : rows_(0), cols_(0) {}
		connection_table(const bool *ar, size_t rows, size_t cols) : connected_(rows * cols), rows_(rows), cols_(cols) {
			std::copy(ar, ar + rows * cols, connected_.begin());
		}

		bool is_connected(size_t x, size_t y) const {
			return is_empty() ? true : connected_[y * cols_ + x];
		}

		bool is_empty() const {
			return rows_ == 0 && cols_ == 0;
		}

		std::vector<bool> connected_;
		size_t rows_;
		size_t cols_;
	};

	enum class padding {
		valid,//use valid pixels of input
		same  //add zero-padding around input so as to keep image size

	};

	template<typename Activation = activation::identity>
	class convolutional_layer :public layer<Activation> {
	public:
		typedef layer<Activation> Base;

		using layer_base::out_size;
		convolutional_layer(cnn_size_t in_width,
			cnn_size_t in_height,
			cnn_size_t window_width,
			cnn_size_t window_height,
			cnn_size_t in_channels,
			cnn_size_t out_channels,
			padding pad_type = padding::valid,
			bool has_bias = true,
			cnn_size_t w_stride = 1,
			cnn_size_t h_stride = 1//in, out weight bias
		) :Base(in_width * in_height * in_channels, conv_out_dim(in_width, in_height, window_width, window_height, w_stride, h_stride, pad_type)*out_channels,
			window_width*window_height*in_channels*out_channels, has_bias ? out_channels : 0),
			in_(in_width, in_height, in_channels),
			in_padded_(in_length(in_width, window_width, pad_type), in_length(in_height, window_height, pad_type), in_channels),
			out_(conv_out_length(in_width, window_width, w_stride, pad_type), conv_out_length(in_height, window_height, h_stride, pad_type), out_channels),
			weight_(window_width, window_height, in_channels*out_channels),
			pad_type_(pad_type),
			w_stride_(w_stride), h_stride_(h_stride) {
			init();
		}

		convolutional_layer(cnn_size_t in_width,
			cnn_size_t in_height,
			cnn_size_t window_size,
			cnn_size_t in_channels,
			cnn_size_t out_channels,
//			const connection_table& connection_table,
			padding pad_type = padding::valid,
			bool has_bias = true,
			cnn_size_t w_stride = 1,
			cnn_size_t h_stride = 1
		)
			: Base(in_width * in_height * in_channels, conv_out_dim(in_width, in_height, window_size, w_stride, h_stride, pad_type) * out_channels,
				sqr(window_size) * in_channels * out_channels, has_bias ? out_channels : 0),
//			tbl_(connection_table),
			in_(in_width, in_height, in_channels),
			in_padded_(in_length(in_width, window_size, pad_type), in_length(in_height, window_size, pad_type), in_channels),
			out_(conv_out_length(in_width, window_size, w_stride, pad_type), conv_out_length(in_height, window_size, h_stride, pad_type), out_channels),
			weight_(window_size, window_size, in_channels*out_channels),
			pad_type_(pad_type),
			w_stride_(w_stride), h_stride_(h_stride)
		{
			init();
		}

		/// number of incoming connections for each output unit
		virtual size_t fan_in_size() const override {
			return weight_.width_ * weight_.height_ * in_.depth_;
		}
		/// number of incoming connections for each output unit
		virtual size_t fan_out_size() const override {
			return (weight_.width_ / w_stride_) * (weight_.height_ / h_stride_) * out_.depth_;
		}
		///  number of connections
		virtual size_t connection_size() const override
		{
			return out_.size() * fan_in_size();
		}

		virtual void forward_propagation(const vec_t& in_raw, size_t worker_index) {
			copy_and_pad_input(in_raw, static_cast<int>(worker_index));

			vec_t &a = a_[worker_index]; // w*x
			vec_t &out = output_[worker_index]; // output
			const vec_t &in = *(prev_out_padded_[worker_index]); // input

			std::fill(a.begin(), a.end(), float_t(0));

			for_i(parallelize_, out_.depth_, [&](int o) {
				for (cnn_size_t inc = 0; inc < in_.depth_; inc++) {
					if (!tbl_.is_connected(o, inc)) continue;

					const float_t *pw = &this->W_[weight_.get_index(0, 0, in_.depth_ * o + inc)];
					const float_t *pi = &in[in_padded_.get_index(0, 0, inc)];
					float_t *pa = &a[out_.get_index(0, 0, o)];

					for (cnn_size_t y = 0; y < out_.height_; y++) {
						for (cnn_size_t x = 0; x < out_.width_; x++) {
							const float_t * ppw = pw;
							const float_t * ppi = pi + (y * h_stride_) * in_padded_.width_ + x * w_stride_;
							float_t sum = float_t(0);

							// should be optimized for small kernel(3x3,5x5)
							for (cnn_size_t wy = 0; wy < weight_.height_; wy++) {
								for (cnn_size_t wx = 0; wx < weight_.width_; wx++) {
									sum += *ppw++ * ppi[wy * in_padded_.width_ + wx];
								}
							}
							pa[y * out_.width_ + x] += sum;
						}
					}
				}

				if (!this->b_.empty()) {
					float_t *pa = &a[out_.get_index(0, 0, o)];
					float_t b = this->b_[o];
					std::for_each(pa, pa + out_.width_ * out_.height_, [&](float_t& f) { f += b; });
				}
			});

			for_i(parallelize_, out_size_, [&](int i) {
				out[i] = h_.f(a, i);
			});
		}
		
		const vec_t& back_propagation(const vec_t& curr_delta, size_t index) override {
			const vec_t& prev_out = *(prev_out_padded_[index]);
			const activation::function& prev_h = prev_->activation_function();
			vec_t* prev_delta = (pad_type_ == padding::same) ? &prev_delta_padded_[index] : &prev_delta_[index];
			vec_t& dW = dW_[index];
			vec_t& db = db_[index];

			std::fill(prev_delta->begin(), prev_delta->end(), float_t(0));

			// propagate delta to previous layer
			for_i(in_.depth_, [&](int inc) {
				for (cnn_size_t outc = 0; outc < out_.depth_; outc++) {
					if (!tbl_.is_connected(outc, inc)) continue;

					const float_t *pw = &this->W_[weight_.get_index(0, 0, in_.depth_ * outc + inc)];
					const float_t *pdelta_src = &curr_delta[out_.get_index(0, 0, outc)];
					float_t *pdelta_dst = &(*prev_delta)[in_padded_.get_index(0, 0, inc)];

					for (cnn_size_t y = 0; y < out_.height_; y++) {
						for (cnn_size_t x = 0; x < out_.width_; x++) {
							const float_t * ppw = pw;
							const float_t ppdelta_src = pdelta_src[y * out_.width_ + x];
							float_t * ppdelta_dst = pdelta_dst + y * h_stride_ * in_padded_.width_ + x * w_stride_;

							for (cnn_size_t wy = 0; wy < weight_.height_; wy++) {
								for (cnn_size_t wx = 0; wx < weight_.width_; wx++) {
									ppdelta_dst[wy * in_padded_.width_ + wx] += *ppw++ * ppdelta_src;
								}
							}
						}
					}
				}
			});

			for_i(parallelize_, in_padded_.size(), [&](int i) {
				(*prev_delta)[i] *= prev_h.df(prev_out[i]);
			});

			// accumulate dw
			for_i(in_.depth_, [&](int inc) {
				for (cnn_size_t outc = 0; outc < out_.depth_; outc++) {

					if (!tbl_.is_connected(outc, inc)) continue;

					for (cnn_size_t wy = 0; wy < weight_.height_; wy++) {
						for (cnn_size_t wx = 0; wx < weight_.width_; wx++) {
							float_t dst = float_t(0);
							const float_t * prevo = &prev_out[in_padded_.get_index(wx, wy, inc)];
							const float_t * delta = &curr_delta[out_.get_index(0, 0, outc)];

							for (cnn_size_t y = 0; y < out_.height_; y++) {
								dst += vectorize::dot(prevo + y * in_padded_.width_, delta + y * out_.width_, out_.width_);
							}
							dW[weight_.get_index(wx, wy, in_.depth_ * outc + inc)] += dst;
						}
					}
				}
			});

			// accumulate db
			if (!db.empty()) {
				for (cnn_size_t outc = 0; outc < out_.depth_; outc++) {
					const float_t *delta = &curr_delta[out_.get_index(0, 0, outc)];
					db[outc] += std::accumulate(delta, delta + out_.width_ * out_.height_, float_t(0));
				}
			}

			if (pad_type_ == padding::same)
				copy_and_unpad_delta(prev_delta_padded_[index], prev_delta_[index]);
			return prev_delta_[index];
		}
		//getter
		index3d<layer_size_t> in_shape() const override { return in_; }
		index3d<layer_size_t> out_shape() const override { return out_; }
		std::string layer_type() const override { return "conv"; } 
		

	public:
		void init() {
			for (cnn_size_t i = 0; i < CNN_QUEUE_SIZE; i++) {
				if (pad_type_ == padding::same) {//
					prev_out_buf_[i] = new vec_t(in_padded_.size(), float_t(0));//bug ==!!!
					prev_delta_padded_[i].resize(in_padded_.size(), float_t(0));
				}
				else {
					prev_out_buf_[i] = nullptr;
				}
			}
		}

		cnn_size_t in_length(cnn_size_t in_length, cnn_size_t window_size, padding pad_type)const {
			return pad_type == padding::same ? (in_length + window_size - 1) : in_length;
		}

		static cnn_size_t conv_out_length(cnn_size_t in_length, cnn_size_t window_size, cnn_size_t stride, padding pad_type) {
			return pad_type == padding::same ? (cnn_size_t)ceil((double)in_length / stride) : (cnn_size_t)ceil((double)(in_length - window_size + 1) / stride);
		}

		static cnn_size_t conv_out_dim(cnn_size_t in_width, cnn_size_t in_height, cnn_size_t window_size, cnn_size_t w_stride, cnn_size_t h_stride, padding pad_type) {
			return conv_out_length(in_width, window_size, w_stride, pad_type) * conv_out_length(in_height, window_size, h_stride, pad_type);
		}

		cnn_size_t conv_out_dim(cnn_size_t in_width, cnn_size_t in_height, cnn_size_t window_width, cnn_size_t window_height, cnn_size_t w_stride, cnn_size_t h_stride, padding pad_type) const {
			return conv_out_length(in_width, window_width, w_stride, pad_type) * conv_out_length(in_height, window_height, h_stride, pad_type);
		}

		void copy_and_unpad_delta(const vec_t& delta, vec_t& dst) {
			if (pad_type_ == padding::valid) {
				dst = delta;
			}
			else {
				for (cnn_size_t c = 0; c < in_.depth_; c++) {
					float_t *pdst = &dst[in_.get_index(0, 0, c)];
					const float_t *pin = &delta[in_padded_.get_index(weight_.width_ / 2, weight_.height_ / 2, c)];

					for (cnn_size_t y = 0; y < in_.height_; y++, pdst += in_.width_, pin += in_padded_.width_) {
						std::copy(pin, pin + in_.width_, pdst);
					}
				}
			}
		}

		
		void copy_and_pad_input(const vec_t& in, int worker_index) {
			vec_t* dst = prev_out_buf_[worker_index];

			if (pad_type_ == padding::valid) {
				prev_out_padded_[worker_index] = &in;
			}
			else {
				// make padded version in order to avoid corner-case in fprop/bprop
				for (cnn_size_t c = 0; c < in_.depth_; c++) {
					float_t *pimg = &(*dst)[in_padded_.get_index(weight_.width_ / 2, weight_.height_ / 2, c)];
					const float_t *pin = &in[in_.get_index(0, 0, c)];

					for (cnn_size_t y = 0; y < in_.height_; y++, pin += in_.width_, pimg += in_padded_.width_) {
						std::copy(pin, pin + in_.width_, pimg);
					}
				}
				prev_out_padded_[worker_index] = prev_out_buf_[worker_index];
			}
		}


		void initIndex(int& outputIndex, int& pre_deltaIndex) {
			if (outputF_[0] == 2) {
				outputIndex = 0;
				pre_deltaIndex = 0;
				not_ready_state();
			}
		}
		bool can_forward(const int& outputIndex, const int& pre_deltaIndex) {
			CNN_UNREFERENCED_PARAMETER(pre_deltaIndex);
			if (outputIndex >= CNN_QUEUE_SIZE) return false;
			if (prev_->outputF_[outputIndex] == 1) return true;
			else return false;
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

				if (layerIndex_ == 1) {
					global_count++;
				//	std::cout << global_count << " ";
				}
			}
		}

		void process() {
			int outputIndex = 0; int pre_deltaIndex = 0;
			while (1) {
				initIndex(outputIndex, pre_deltaIndex);
				if (can_forward(outputIndex, pre_deltaIndex)) {
					forward_propagation(prev_->output_[outputIndex], outputIndex);
					outputF_[outputIndex] = 1;
					//	if (outputIndex == CNN_QUEUE_SIZE - 1) { std::cout << "conv1 forward finished"<<std::endl; }
					outputIndex++;

				}

				if (next_) {
					if (can_backward(outputIndex, pre_deltaIndex)) {
						//backward process
						back_propagation(next_->prev_delta_[pre_deltaIndex], pre_deltaIndex);
						//if (pre_deltaIndex == CNN_QUEUE_SIZE - 1) { std::cout << "conv backward finished" << std::endl; }
						prev_deltaF_[pre_deltaIndex] = 1;
						pre_deltaIndex++;
					}
				}
				else {//the last fully connected
					if (can_backward(outputIndex, pre_deltaIndex)) {
						back_propagation(current_delta_[pre_deltaIndex], pre_deltaIndex);
						//if (pre_deltaIndex == CNN_QUEUE_SIZE - 1) { std::cout << "last_conv backward finished" << std::endl; }
						prev_deltaF_[pre_deltaIndex] = 1;
						pre_deltaIndex++;
					}
				}
			}
		}
	public:
		const vec_t* prev_out_padded_[CNN_QUEUE_SIZE];
		vec_t* prev_out_buf_[CNN_QUEUE_SIZE];
		vec_t prev_delta_padded_[CNN_QUEUE_SIZE];

		connection_table tbl_;
		index3d<layer_size_t> in_;
		index3d<layer_size_t> in_padded_;
		index3d<layer_size_t> out_;
		index3d<layer_size_t> weight_;
		padding pad_type_;
		size_t w_stride_;
		size_t h_stride_;
 

	};


#if 0
	template<typename Activation = activation::identity>
	class convolutional_layer :public partial_connected_layer<Activation> {
	public:
		typedef partial_connected_layer<Activation> Base;
		convolutional_layer(layer_size_t in_width, layer_size_t in_height, layer_size_t window_size, layer_size_t in_channels, layer_size_t out_channels)
			:Base(in_width * in_height * in_channels, (in_width - window_size + 1) * (in_height - window_size + 1) * out_channels,
				sqr(window_size) * in_channels * out_channels, out_channels),
			in_(in_width, in_height, in_channels),
			out_((in_width - window_size + 1), (in_height - window_size + 1), out_channels),
			weight_(window_size, window_size, in_channels*out_channels),
			window_size_(window_size) {
			init_connection(connection_table()); 
		}

		convolutional_layer(layer_size_t in_width, layer_size_t in_height, layer_size_t window_size, layer_size_t in_channels, layer_size_t out_channels, const connection_table& connection_table)
			: Base(in_width * in_height * in_channels, (in_width - window_size + 1) * (in_height - window_size + 1) * out_channels,
				sqr(window_size) * in_channels * out_channels, out_channels),
			in_(in_width, in_height, in_channels),
			out_((in_width - window_size + 1), (in_height - window_size + 1), out_channels),
			weight_(window_size, window_size, in_channels*out_channels),
			connection_(connection_table),
			window_size_(window_size) {
			init_connection(connection_table);
			//this->remap();
		}

		index3d<layer_size_t> in_shape() const override { return in_; }
		index3d<layer_size_t> out_shape() const override { return out_; }
		std::string layer_type() const override { return "conv"; }



	private:
		void init_connection(const connection_table& table) {
			for (layer_size_t inc = 0; inc < in_.depth_; ++inc) {
				for (layer_size_t outc = 0; outc < out_.depth_; ++outc) {
					if (!table.is_connected(outc, inc)) { continue; }
					for (layer_size_t y = 0; y < out_.height_; ++y)
						for (layer_size_t x = 0; x < out_.width_; ++x)
							connect_kernel(inc, outc, x, y);
				}
			}

			for (layer_size_t outc = 0; outc < out_.depth_; ++outc)
				for (layer_size_t y = 0; y < out_.height_; ++y)
					for (layer_size_t x = 0; x < out_.width_; ++x)
						this->connect_bias(outc, out_.get_index(x, y, outc));
		}

		void connect_kernel(layer_size_t inc, layer_size_t outc, layer_size_t x, layer_size_t y) {
			for (layer_size_t dy = 0; dy < window_size_; ++dy)
				for (layer_size_t dx = 0; dx < window_size_; ++dx)
					this->connect_weight(
						in_.get_index(x + dx, y + dy, inc),
						out_.get_index(x, y, outc),
						weight_.get_index(dx, dy, outc * in_.depth_ + inc));
		}




		index3d<layer_size_t> in_;
		index3d<layer_size_t> out_;
		index3d<layer_size_t> weight_;
		connection_table connection_;
		size_t window_size_;
	};
#endif
}
