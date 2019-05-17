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

	template<typename Activation, typename Filter = filter_none>
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
}