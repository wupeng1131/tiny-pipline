#pragma once
#include "util.h"

namespace tiny_cnn {
namespace weight_init {
	class function {
	public:
		virtual void fill(vec_t *weight, layer_size_t fan_in, layer_size_t fan_out) = 0;
		virtual function* clone() const = 0;
	};

	class scalable : public function {
	public:
		scalable(float_t value) : scale_(value) {}

		void scale(float_t value) {
			scale_ = value;
		}
	protected:
		float_t scale_;
	};
	
	/*
	*use fan-in and fan-out for scaling
	*understanding the difficulty   of training deep feedforward neural networks
	*/
	class xavier : public scalable {
	public:
		xavier() : scalable((float_t)6.0) {}
		explicit xavier(float_t value) : scalable(value) {}
		void fill(vec_t *weight, layer_size_t fan_in, layer_size_t fan_out) override {
			const float_t weight_base = std::sqrt(scale_ / (fan_in + fan_out));
			uniform_rand(weight->begin(), weight->end(), -weight_base, weight_base);
		}

		virtual xavier* clone() const override { return new xavier(scale_); }

	};

	/*use fan-in for scaling*/
	class lecun : public scalable {
	public:
		lecun() : scalable((float_t)1.0) {}
		explicit lecun(float_t value) : scalable(value) {}
		
		void fill(vec_t *weight, layer_size_t fan_in, layer_size_t fan_out) {
			CNN_UNREFERENCED_PARAMETER(fan_out);
			const float_t weight_base = scale_ / std::sqrt(fan_in);
			uniform_rand(weight->begin(), weight->end(), -weight_base, weight_base);
		}

		virtual lecun* clone() const override { return new lecun(scale_); }

	};

	class constant : public scalable {
	public:
		constant() : scalable((float_t)0.0) {}
		explicit constant(float_t value) : scalable(value){}

		void fill(vec_t *weight, layer_size_t fan_in, layer_size_t fan_out) {
			CNN_UNREFERENCED_PARAMETER(fan_in);
			CNN_UNREFERENCED_PARAMETER(fan_out);
			std::fill(weight->begin(), weight->end(), scale_);
		}
		virtual constant* clone() const override { return new constant(scale_); }

	};
}//namespace weight_init

}//namespace tiny_cnn
