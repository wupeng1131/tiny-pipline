#pragma once
#include "layer.h"
#include "util.h"
namespace tiny_cnn {

	class input_layer : public layer< activation::identity > {
	public:
		typedef activation::identity Activation;
		typedef layer<activation::identity> Base;
		//CNN_USE_LAYER_MEMBERS;
		input_layer() : Base(0, 0, 0, 0) {
			for (int i = 0; i < CNN_QUEUE_SIZE; i++)  test[i] = 0;
		}
		
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
		}

		const vec_t& back_propagation(const vec_t& current_delta, size_t index) {
			
			return current_delta;
		}

		/***********************************************/
		void f_process()override {}
		void b_process()override {}
		void process() override {}
		/*
		void initIndex(int& outputIndex, int& pre_deltaIndex) {
			if (outputF_[0] == 2) { 
				pre_deltaIndex = 0; 
				outputIndex = 0; 
				not_ready_state();
				for (int i = 0; i < CNN_QUEUE_SIZE; i++)  test[i] = 0;
			}//updateing state, reset the index
		}

		bool can_forward(const int& outputIndex, const int& pre_deltaIndex) {
			CNN_UNREFERENCED_PARAMETER(pre_deltaIndex);
			if (outputIndex >= CNN_QUEUE_SIZE) return false;
			else return false;//not compute and not full
		}
		bool can_backward(const int& outputIndex, const int& pre_deltaIndex) {
			CNN_UNREFERENCED_PARAMETER(outputIndex);
			if (pre_deltaIndex >= CNN_QUEUE_SIZE) { 
				//std::cout << pre_deltaIndex; 
				return false; }
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
		void f_process()override {//need to be in a while
			
			initIndex(outputIndex_, pre_deltaIndex_);//no need to forward, just init
			
			
		}
		void b_process()override {
			
			initIndex(outputIndex_, pre_deltaIndex_);
			if (can_backward(outputIndex_, pre_deltaIndex_)) {
				
				back_propagation(next_->prev_delta_[pre_deltaIndex_], pre_deltaIndex_);
				current_deltaF_[pre_deltaIndex_] = 1;
				prev_deltaF_[pre_deltaIndex_] = 1;//bug  ignore this operate
				test[pre_deltaIndex_] = pre_deltaIndex_;
				pre_deltaIndex_++;
			}
		}*/
		/*void process() override {
			
			int outputIndex = 0; int pre_deltaIndex = 0;
			//int flag = 0;
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
		}*/
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
