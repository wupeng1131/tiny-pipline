#pragma once

#include"util.h"
#include <unordered_map>

namespace tiny_cnn {
	template<bool usesHessian>
	struct optimizer {
		bool requires_hessian() const { return usesHessian; }
		virtual void reset() {}// override to implement pre-learning action
	};

	//helper class to hold N values for each weight
	template<typename value_t, int N, bool usesHessian = false>
	struct stateful_optimizer :public optimizer<usesHessian> {
		void reset() override {
			for (auto& e : E_) e.clear();
		}

	protected:
		template <int Index>
		std::vector<value_t>& get(const vec_t& key) {
			static_assert(Index < N, "index out of range");
			if(E_[Index][&key].empty())
				E_[Index][&key].resize(key.size(), value_t());
			return E_[Index][&key];
		}

		std::unordered_map<const vec_t*, std::vector<value_t>> E_[N];
	};
	//adagrad
	struct adagrad : public stateful_optimizer<float_t, 1, false> {
		adagrad() :alpha(0.01), eps(1e-8) {}
		void update(const vec_t& dW, const vec_t&/*hessian*/, vec_t &W) {
			vec_t& g = get<0>(W);

			for_i(W.size(), [&](int i) {
				g[i] += dW[i] * dW[i];
				W[i] -= alpha * dW[i] / (std::sqrt(g[i]) + eps);
			});
		}
		float_t alpha;
	private:
		float_t eps;
	};

	//RMSprop
	struct RMSprop :public stateful_optimizer<float_t, 1, false> {
		RMSprop() :alpha(0.0001), mu(0.99), eps(1e-8) {}
		
		void update(const vec_t& dW, const vec_t&/*hessian*/, vec_t& W) {
			vec_t& g = get<0>(W);

			for_i(W.size(), [&](int i) {
				g[i] = mu*g[i] + (1 - mu) * dW[i] * dW[i];
				W[i] -= alpha * dW[i] / std::sqrt(g[i] + eps);
			});
		}

		float_t alpha;
		float_t mu;
	private:
		float_t eps;
	};

	//adam
	struct Adam :public stateful_optimizer<float_t, 2, false> {
		Adam() :alpha(0.001), b1(0.9), b2(0.999), b1_t(0.9), b2_t(0.999), eps(1e-8) {}

		void update(const vec_t& dW, const vec_t& /*hessian*/,vec_t & W) {
			vec_t& mt = get<0>(W);
			vec_t& vt = get<1>(W);

			b1_t *= b1; b2_t *= b2;
			for_i(W.size(), [&](int i) {
				mt[i] = b1 * mt[i] + (1 - b1) * dW[i];
				vt[i] = b2 * vt[i] + (1 - b2) * dW[i] * dW[i];

				W[i] -= alpha * (mt[i] / (1 - b1_t)) / std::sqrt((vt[i] / (1 - b2_t)) + eps);
			});

		}

		float_t alpha;//learning rate
		float_t b1;//decay term
		float_t b2;//decay term
		float_t b1_t;//decay term power t
		float_t b2_t;//decay term power t
	private:
		float_t eps;// constant value to avoid aero-division

	};

	//sgd
	struct gradient_denscent :public optimizer<false> {
		gradient_denscent() :alpha(0.01), lambda(0.0) {}

		void update(const vec_t& dW, const vec_t& /*hessian*/, vec_t& W) {
			for_i(W.size(), [&](int i) {
				W[i] = W[i] - alpha * (dW[i] + lambda * W[i]);
			});
		}
		float_t alpha;
		float_t lambda;
	};

	//momentum
	struct momentum :public stateful_optimizer<float_t, 1, false> {
	public:
		momentum() {}
		void update(const vec_t& dW, const vec_t& /*hessian*/, vec_t& W) {
			vec_t& dWprev = get<0>(W);
			
			for_i(W.size(), [&](int i) {
				float_t V = mu * dWprev[i] - alpha * (dW[i] + W[i] * lambda);
				W[i] += V;
				dWprev[i] = V;
			});
		}

		float_t alpha;
		float_t lambda;
		float_t mu;

	};

}
