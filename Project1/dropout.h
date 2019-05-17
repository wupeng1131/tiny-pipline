#pragma once
#include "layer.h"
#include "product.h"
#include "util.h"

namespace tiny_cnn {
	
	class filter_none {
	public:
		explicit filter_none(int out_dim) {
			CNN_UNREFERENCED_PARAMETER(out_dim);
		}
		const vec_t& filter_fprop(const vec_t& out, int index) {
			CNN_UNREFERENCED_PARAMETER(index);
			return out;
		}

		const vec_t& filter_bprop(const vec_t& delta, int index) {
			CNN_UNREFERENCED_PARAMETER(index);
			return delta;
		}
	};

	class dropout {
	public:
		enum context {
				train_phase,
				test_phase
		};
		enum mode {
			per_data,
			per_batch
		};
		explicit dropout(int out_dim)
			:out_dim_(out_dim), mask_(out_dim), ctx_(train_phase), mode_(per_data),dropout_rate_(0.5) {
			for (int i = 0; i < CNN_QUEUE_SIZE; i++) {
				masked_out_[i].resize(out_dim);
				masked_delta_[i].resize(out_dim);
			}
			shuffle();
		}

		void shuffle() {
			for (auto& m : mask_) {
				m = bernoulli(1.0 - dropout_rate_);
			}
		}

		void set_dropout_rate(double rate) {
			if (rate < 0.0 || rate >= 1.0)
				throw nn_error("dropout-rate error");
		}

		void set_mode(mode mode) {
			mode_ = mode;
		}

		void set_context(context ctx) {
			ctx_ = ctx;
		}

		const vec_t& filter_fprop(const vec_t& out, int index) {
			if (ctx_ == train_phase) {
				for (int i = 0; i < out_dim_; i++) {
					masked_out_[index][i] = out[i] * mask_[i];
				}
			}
			else if (ctx_ == test_phase) {
				for (int i = 0; i < out_dim_; i++) {
					masked_out_[index][i] = out[i] * (1.0 - dropout_rate_);
				}
			}
			else { throw nn_error("invalid context"); }
			return masked_out_[index];
		}

		//mask delta
		const vec_t& filter_bprop(const vec_t& delta, int index) {
			for (int i = 0; i < out_dim_; i++)
				masked_delta_[index][i] = delta[i] * mask_[i];
			if (mode_ == per_data)  shuffle();
			return masked_delta_[index];
		}

		void end_batch() {
			if (mode_ == per_batch) shuffle();
		}

	private:
		int out_dim_;
		std::vector<uint8_t> mask_;
		vec_t masked_out_[CNN_QUEUE_SIZE];
		vec_t masked_delta_[CNN_QUEUE_SIZE];
		context ctx_;
		mode mode_;
		double dropout_rate_;


	};
}