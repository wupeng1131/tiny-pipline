#pragma once
#include "util.h"
namespace tiny_cnn {
//for regression
class mse{
public:
	static float_t  f(float_t y, float_t t) {
		return(y - t) * (y - t) / 2;
	}
	
	static float_t df(float_t y, float_t t) {
		return y - t;
	}
};

class cross_entropy {
public:
	static float_t f(float_t y, float_t t) {
		return -t * std::log(y) - (1.0 - t) * std::log(1 - y);
	}

	static float_t df(float_t y, float_t t) {
		return (y - t) / (y * (1 - y));
	}
};

class cross_entropy_multiclass {
public:
	static float_t f(float_t y, float_t t) {
		return -t * std::log(y);
	}

	static float_t df(float_t y, float_t t) {
		return -t / y;
	}
};

template <typename E>
vec_t gradient(const vec_t& y, const vec_t& t) {
	vec_t grad(y.size());
	assert(y.size() == t.size());

	for (size_t i = 0; i < y.size(); i++) {
		grad[i] = E::df(y[i], t[i]);
	}

	return grad;
}

}
