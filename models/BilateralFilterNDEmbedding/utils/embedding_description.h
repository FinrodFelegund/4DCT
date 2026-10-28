#include <torch/extension.h>

//Struct to easily cache descriptive information about an embedding feed forward network.
//Helpful, since regular calls to member function of torch::Tensor cause memory issues

struct EmbeddingDescription {
    public:
        EmbeddingDescription(torch::Tensor in_layer, torch::Tensor hidden_layer, torch::Tensor out_layer){
            dimensions = hidden_layer.dim();

            for(int i = 0; i < dimensions; i++){
                in_sizes[i] = in_layer.size(i);
                in_strides[i] = in_layer.stride(i);

                hidden_sizes[i] = hidden_layer.size(i);
                hidden_strides[i] = hidden_layer.stride(i);

                out_sizes[i] = out_layer.size(i);
                out_strides[i] = out_layer.stride(i);
            }
    
        }

        ~EmbeddingDescription(){

        }

        int dimensions;

        int in_sizes[3];
        int in_strides[3];

        int hidden_sizes[3];
        int hidden_strides[3];

        int out_sizes[3];
        int out_strides[3];
};