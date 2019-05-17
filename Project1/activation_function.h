#pragma once
#include "util.h"
#include <algorithm>

namespace tiny_cnn {
namespace activation {
	class function {
	public:
		virtual ~function() {}
		virtual float_t f(const vec_t& v, size_t index) const = 0;

		// dfi/dyi
		virtual float_t df(float_t y) const = 0;

		// dfi/ dyk (k=0,1,...n)
		virtual vec_t df(const vec_t& y, size_t i)const {
			vec_t v(y.size(), 0);
			v[i] = df(y[i]);
			return v;
		}

		//target value range for learning
		virtual std::pair<float_t, float_t>scale() const = 0;

	};

	class identity : public function {
	public:
		float_t f(const vec_t& v, size_t i) const override { return v[i]; }
		float_t df(float_t /*y*/) const override { return 1; }
		std::pair<float_t, float_t> scale() const override { return std::make_pair(0.1, 0.9); }
	};


	class sigmoid : public function {
	public:
		float_t f(const vec_t& v, size_t i) const override { return 1.0 / (1.0 + std::exp(-v[i])); }
		float_t df(float_t y) const override { return y * (1.0 - y); }
		std::pair<float_t, float_t> scale() const override { return std::make_pair(0.1, 0.9); }
	};

	class relu : public function {
	public:
		float_t f(const vec_t& v, size_t i) const override { return max(static_cast<float_t>(0.0), v[i]); }
		float_t df(float_t y) const override { return y > 0.0 ? 1.0 : 0.0; }
		std::pair<float_t, float_t> scale() const override { return std::make_pair(0.1, 0.9); }
	};

	//typedef relu rectified_linear; //for compatibility
	

	class leaky_relu : public function {
	public:
		float_t f(const vec_t& v, size_t i) const override { return (v[i] > 0) ? v[i] : 0.01 * v[i]; }
		float_t df(float_t y) const override { return y > 0.0 ? 1.0 : 0.01; }
		std::pair<float_t, float_t> scale() const override { return std::make_pair(0.1, 0.9); }
	};

	class softmax : public function {
	public:
		float_t f(const vec_t& v, size_t i) const override {
			float_t alpha = *std::max_element(v.begin(), v.end());
			float_t numer = std::exp(v[i] - alpha);
			float_t denom = 0.0;
			for (auto x : v) {
				denom += std::exp(x - alpha);
			}
			return numer / denom;
		}

		float_t df(float_t y)const override {
			return y * (1.0 - y);
		}

		virtual vec_t df(const vec_t& y, size_t index) const override {
			vec_t v(y.size(), 0);
			for (size_t i = 0; i < y.size(); i++) {
				v[i] = (i == index) ? df(y[index]) : -y[i] * y[index];
				}
			return v;
		}

		std::pair<float_t, float_t> scale() const override {
			return std::make_pair(0.0, 1.0);
		}
	};

	class tan_h : public function {
	public:
		float_t f(const vec_t& v, size_t i) const override {
			const float_t ep = std::exp(v[i]);
			const float_t em = std::exp(-v[i]);
			return (ep - em) / (ep + em);
		}

		float_t df(float_t y) const override { return 1.0 - sqr(y); }
		std::pair<float_t, float_t> scale() const override { return std::make_pair(-0.8, 0.8); }

	};

	// s tan_h, but scaled to match the other functions
	class tan_hp1m2 : public function {
	public:
		float_t f(const vec_t& v, size_t i) const override {
			const float_t ep = std::exp(v[i]);
			return ep / (ep + std::exp(-v[i]));
		}

		float_t df(float_t y) const override { return 2 * y *(1.0 - y); }
		std::pair<float_t, float_t> scale() const override { return std::make_pair(0.1, 0.9); }
	};

}
}
