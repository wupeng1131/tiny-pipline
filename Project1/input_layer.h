#pragma once
#include "layer.h"
#include "util.h"
namespace tiny_cnn {

	class input_layer : public layer< activation::identity > {
	public:
		typedef activation::identity Activation;
		typedef layer<activation::identity> Base;
		//CNN_USE_LAYER_MEMBERS;
		input_layer() : Base(0, 0, 0, 0) {}
		
		layer_size_t in_size() const override {
			return next_ ? next_->in_size() : static_cast<layer_size_t>(0);
		}
		index3d<layer_size_t> in_shape() const override {
			return next_ ? next_->in_shape() : index3d<layer_size_t>(0, 0, 0);
		}
		index3d<layer_size_t> out_shape() const override { 
			return next_ ? next_->out_shape() : index3d<layer_size_t>(0, 0, 0); 
		}
		std::string layer_type() const override { 
			return next_ ? next_->layer_type() : "input"; 
		}
		//const vec_t& forward_propagation(size_t index);//use output to put the in_data
		 void forward_propagation(const vec_t& in, size_t index) {
			output_[index] = in;
			//return next_ ? next_->forward_propagation(in, index) : output_[index];
			//return output_[index];
		}

		const vec_t& back_propagation(const vec_t& current_delta, size_t index) {
			
			return current_delta;
		}

		/***********************************************/
		void initIndex(int& outputIndex, int& pre_deltaIndex) {
			if (outputF_[0] == 2) { outputIndex = 0; pre_deltaIndex = 0; not_ready_state();
			}//updateing state, reset the index
		}

		bool can_forward(const int& outputIndex, const int& pre_deltaIndex) {
			CNN_UNREFERENCED_PARAMETER(pre_deltaIndex);
			if (outputIndex >= CNN_QUEUE_SIZE) return false;
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
		void process() override {
			
			int outputIndex = 0; int pre_deltaIndex = 0;
			int flag = 0;
			while(1) {
				initIndex(outputIndex, pre_deltaIndex);
				//if (outputF_[CNN_QUEUE_SIZE - 1] == 1 && flag == 0) { std::cout << "input forward finished"<<std::endl; flag++; }
					
				if (can_backward(outputIndex, pre_deltaIndex)) {
					back_propagation(next_->prev_delta_[pre_deltaIndex], pre_deltaIndex);
				//	if (pre_deltaIndex == CNN_QUEUE_SIZE - 1) { std::cout << "input backward finished" << std::endl; }
					current_deltaF_[pre_deltaIndex] = 1;
					pre_deltaIndex++;
				}
			}
		}
		/*********************************************/

		//getter
		size_t connection_size() const override {
			return in_size_;
		}
		size_t fan_in_size() const override {
			return 1;
		}
		size_t fan_out_size() const override {
			return 1;
		}
	};

}//namespace tiny_cnn
