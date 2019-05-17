#pragma once
#include "input_layer.h"

namespace tiny_cnn {

	class layers {
	public:
		layers() { add(std::make_shared<input_layer>()); }
		layers(const layers& rhs) { construct(rhs); }
		layers& operator = (const layers& rhs) {
			layers_.clear();
			construct(rhs);
			return *this;
		}
		/****   input new_tail to layers    ********/
		void add(std::shared_ptr<layer_base> new_tail) {
			if (tail()) tail()->connect(new_tail);
			layers_.push_back(new_tail);
		}

		//getter
		size_t depth() const {
			return layers_.size() - 1;
		}
		bool empty() const { return layers_.size() == 0; }
		layer_base* head() const { return empty() ? 0 : layers_[0].get(); }
		layer_base* tail() const { return empty() ? 0 : layers_[layers_.size() - 1].get(); }

		template <typename T>
		const T& at(size_t index) const {
			const T* v = dynamic_cast<const T*>(layers_[index + 1].get());
			if (v) return *v;
			throw nn_error("failed to cast");
		}

		const layer_base* operator[](size_t index) const {
			return layers_[index + 1].get();
		}

		layer_base* operator[] (size_t index) {
			return layers_[index + 1].get();
		}

		void init_weight() {
			for (auto p1 : layers_) {
				p1->init_weight();
			}
		}

		bool is_exploded() const {
			for (auto pl : layers_) {
				if (pl->is_exploded()) return true;
			}
			return false;
		}

		template <typename Optimizer>
		void update_weights(Optimizer *o, size_t worker_size, size_t batch_size) {
			for (auto pl : layers_) {
				pl->update_weight(o, worker_size, batch_size);
			}
		}

		void set_parallelize(bool parallelize) {
			for (auto pl : layers_) {
				pl->set_parallelize(parallelize);
			}
		}

	public:
		void construct(const layers& rhs) {
			add(std::make_shared<input_layer>());
			for (size_t i = 1; i < rhs.layers_.size(); i++)
				add(rhs.layers_[i]);
		}

		std::vector<std::shared_ptr<layer_base>> layers_;
	};

}//namespace tiny_cnn
