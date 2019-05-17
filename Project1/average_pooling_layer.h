#pragma once

#include "util.h"
#include "partial_connected_layer.h"
//#include "image.h"

namespace tiny_cnn {
	
	template<typename Activation>
	class average_pooling_layer : public partial_connected_layer<Activation> {
	public:
		typedef partial_connected_layer<Activation> Base;

		average_pooling_layer(layer_size_t in_width, layer_size_t in_height, layer_size_t in_channels, layer_size_t pooling_size)
			: Base(in_width * in_height * in_channels,//in
				in_width * in_height * in_channels / sqr(pooling_size),//out
				in_channels, in_channels, 1.0 / sqr(pooling_size)),
			in_(in_width , in_height, in_channels),
			out_(in_width / pooling_size, in_height / pooling_size, in_channels) {
			if ((in_width % pooling_size) || (in_height % pooling_size))
				pooling_size_mismatch(in_width, in_height, pooling_size);

			init_connection(pooling_size);
		}

		index3d<layer_size_t> in_shape() const override { return in_; }
		index3d<layer_size_t> out_shape() const override { return out_; }
		std::string layer_type() const override { return "ave-pool"; }

	private:
		void init_connection(layer_size_t pooling_size) {
			for (layer_size_t c = 0; c < in_.depth_; ++c)
				for (layer_size_t y = 0; y < in_.height_; y += pooling_size)
					for (layer_size_t x = 0; x < in_.width_; x += pooling_size)
						connect_kernel(pooling_size, x, y, c);


			for (layer_size_t c = 0; c < in_.depth_; ++c)
				for (layer_size_t y = 0; y < out_.height_; ++y)
					for (layer_size_t x = 0; x < out_.width_; ++x)
						this->connect_bias(c, out_.get_index(x, y, c));
		}

		void connect_kernel(layer_size_t pooling_size, layer_size_t x, layer_size_t y, layer_size_t inc) {
			for (layer_size_t dy = 0; dy < pooling_size; ++dy)
				for (layer_size_t dx = 0; dx < pooling_size; ++dx)
					this->connect_weight(
						in_.get_index(x + dx, y + dy, inc),
						out_.get_index(x / pooling_size, y / pooling_size, inc),
						inc);
		}

		index3d<layer_size_t> in_;
		index3d<layer_size_t> out_;
	};

}