#pragma once

#include "util.h"
#include "layer.h"

namespace tiny_cnn {
	template<typename Activation>
	class partial_connected_layer : public layer<Activation> {
	public: 
		typedef std::vector<std::pair<unsigned short, unsigned short>> io_connections;
		typedef std::vector<std::pair<unsigned short, unsigned short>> wi_connections;
		typedef std::vector<std::pair<unsigned short, unsigned short>> wo_connections;
		typedef layer<Activation> Base;
		
		partial_connected_layer(layer_size_t in_dim, layer_size_t out_dim, size_t weight_dim, size_t bias_dim, float_t scale_factor = 1.0)
			: Base(in_dim, out_dim, weight_dim, bias_dim),
			weight2io_(weight_dim), out2wi_(out_dim), in2wo_(in_dim), bias2out_(bias_dim), out2bias_(out_dim),
			scale_factor_(scale_factor) {}

		size_t param_size() const override {
			size_t total_param = 0;
			for (auto w : weight2io_)
				if (w.size() > 0) total_param++;
			for (auto b : bias2out_)
				if (b.size() > 0) total_param++;
			return total_param;
		}

		size_t connection_size() const override {    //is it the real num of param
			size_t total_size = 0;
			for (auto io : weight2io_)
				total_size += io.size();
			for (auto b : bias2out_)
				total_size += b.size();
			return total_size;
		}

		size_t fan_in_size() const override {       //?why do it??
			return max_size(out2wi_);

		}

		size_t fan_out_size() const override {
			return max_size(in2wo_);
		}

		void connect_weight(layer_size_t input_index, layer_size_t output_index, layer_size_t weight_index) {
			weight2io_[weight_index].emplace_back(input_index, output_index);
			out2wi_[output_index].emplace_back(weight_index, input_index);
			in2wo_[input_index].emplace_back(weight_index, output_index);
		}

		void connect_bias(layer_size_t bias_index, layer_size_t output_index) {
			out2bias_[output_index] = bias_index;//a out could related to a bias
			bias2out_[bias_index].push_back(output_index);// a bias could related to many out
		}
		/*forward    
		*input: input data <in>   <index>
		*output: the result
		*/
		 void forward_propagation(const vec_t& in, size_t index)override {
			vec_t&a = a_[index];	//a is a out

			for_i(parallelize_, out_size_, [&](int i) {	//out is a vector
				const wi_connections& connections = out2wi_[i];//each output has a  weight in relation
				a[i] = 0.0;	//a unit
				for (auto connection : connections) {
					a[i] += W_[connection.first] * in[connection.second];
				}

				a[i] *= scale_factor_;
				a[i] += b_[out2bias_[i]];
			});

			for_i(parallelize_, out_size_, [&](int i) {
				output_[index][i] = h_.f(a, i); 
			});
			//return next_ ? next_->forward_propagation(output_[index], index) : output_[index];
			//return output_[index];
		}

		virtual const vec_t&back_propagation(const vec_t& current_delta, size_t index) {
			const vec_t& prev_out = prev_->output(index);
			const activation::function& prev_h = prev_->activation_function();
			vec_t& prev_delta = prev_delta_[index];

			for_(parallelize_, 0, in_size_, [&](const blocked_range& r) {
				for (int i = r.begin(); i != r.end(); i++) {	//for each input
					const wo_connections& connections = in2wo_[i];// wo->in
					float_t delta = 0.0;

					for (auto connection : connections) {//each input has w-o connections 
						delta += W_[connection.first] * current_delta[connection.second];//delta of layer
					}

					prev_delta[i] = delta * scale_factor_ * prev_h.df(prev_out[i]);
				}
			});

			for_(parallelize_, 0, weight2io_.size(), [&](const blocked_range& r) {
				for (int i = r.begin(); i < r.end(); i++) {//for all kernel unit
					const io_connections& connections = weight2io_[i];//i st kernel
					float_t diff = 0.0;

					for (auto connection : connections) {
						diff += prev_out[connection.first] * current_delta[connection.second];
					}

					dW_[index][i] += diff * scale_factor_;//why use +=	  ???
				}
			});

			for (size_t i = 0; i < bias2out_.size(); i++) {// n kernel has n bias
				const std::vector<layer_size_t>& outs = bias2out_[i];
				float_t diff = 0.0;

				for (auto o : outs) { 
					diff += current_delta[o];
				}
				db_[index][i] += diff;
			}
			return prev_delta_[index];// prev_->back_propagation(prev_delta_[index], index);
		}

		/**************process**************/
		void initIndex(int& outputIndex, int& pre_deltaIndex) {
			if (outputF_[0] == 2) { outputIndex = 0; pre_deltaIndex = 0; not_ready_state();
			}//updateing state, reset the index
		}
		bool can_forward(const int& outputIndex, const int& pre_deltaIndex) {
			CNN_UNREFERENCED_PARAMETER(pre_deltaIndex);
			if (outputIndex >= CNN_QUEUE_SIZE) return false;//bug: index overflow
			if (prev_->outputF_[outputIndex] == 1 ) return true;
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
		void process() {  
			int outputIndex = 0; int pre_deltaIndex = 0;
			while (1) {
				initIndex(outputIndex, pre_deltaIndex);
				if (can_forward(outputIndex, pre_deltaIndex)) {
					forward_propagation(prev_->output_[outputIndex], outputIndex);
					outputF_[outputIndex] = 1;
				//	if (outputIndex == CNN_QUEUE_SIZE - 1) { std::cout << "conv1 forward finished"<<std::endl; }
					outputIndex++;
				
				}

				if (next_) {
					if (can_backward(outputIndex, pre_deltaIndex)) {
						//backward process
						back_propagation(next_->prev_delta_[pre_deltaIndex], pre_deltaIndex);
						//if (pre_deltaIndex == CNN_QUEUE_SIZE - 1) { std::cout << "conv backward finished" << std::endl; }
						prev_deltaF_[pre_deltaIndex] = 1;			
						pre_deltaIndex++;
					}
				}
				else {//the last fully connected
					if (can_backward(outputIndex, pre_deltaIndex)) {
						back_propagation(current_delta_[pre_deltaIndex], pre_deltaIndex);
						//if (pre_deltaIndex == CNN_QUEUE_SIZE - 1) { std::cout << "last_conv backward finished" << std::endl; }
						prev_deltaF_[pre_deltaIndex] = 1;
						pre_deltaIndex++;
					}
				}			
			}
		}
		/**********process*********/

	protected:
		std::vector<io_connections> weight2io_;
		std::vector<wi_connections> out2wi_;
		std::vector<wo_connections> in2wo_;
		std::vector<std::vector<layer_size_t>> bias2out_;
		std::vector<size_t> out2bias_;
		float_t scale_factor_;


	};


}
