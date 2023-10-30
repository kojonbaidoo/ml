#include "nn.h"

typedef struct
{
    size_t neurons;
    Matrix weights;
    Matrix bias;
    Matrix output;
    Matrix state;
    Matrix s_weights;
    int activation;
} E_RNN_Layer;

typedef struct{
    E_RNN_Layer *layers;
    size_t num_layers;
    size_t max_num_layers;
} E_RNN;

E_RNN_Layer e_rnn_layer_alloc(size_t num_inputs, size_t num_neurons, int activation);

E_RNN e_rnn_alloc(size_t num_layers);
void e_rnn_add(E_RNN *rnn, E_RNN_Layer layer);
void e_rnn_forward(E_RNN rnn, Matrix input);
void e_rnn_train(E_RNN rnn, Matrix td_x, Matrix td_y, float lr, size_t epochs);
void e_rnn_backprop(E_RNN rnn, Matrix td_x, Matrix td_y, float lr);
E_RNN e_rnn_copy(E_RNN rnn);

#ifndef RNN_H_
void e_rnn_layer_free(E_RNN_Layer layer);
void e_rnn_free(E_RNN mlp);

#define RNN_H_

E_RNN e_rnn_alloc(size_t num_layers){
    E_RNN rnn;
    rnn.num_layers = 0;
    rnn.max_num_layers = num_layers;
    rnn.layers = malloc(num_layers * sizeof(E_RNN_Layer));

    return rnn;
}

E_RNN_Layer e_rnn_layer_alloc(size_t num_inputs, size_t num_neurons, int activation){
    E_RNN_Layer layer;
    layer.neurons = num_neurons;
    layer.weights = mat_alloc(num_neurons, num_inputs);
    layer.s_weights = mat_alloc(num_neurons, num_neurons);
    layer.bias = mat_alloc(num_neurons, 1);
    layer.output = mat_alloc(num_neurons, 1);
    layer.state = mat_alloc(num_neurons, 1);
    layer.activation = activation;

    mat_rand(layer.weights);
    mat_rand(layer.s_weights);
    mat_rand(layer.bias);
    mat_fill(layer.output, 0);
    mat_fill(layer.state, 0);
    return layer;
}

void e_rnn_add(E_RNN *rnn, E_RNN_Layer layer){
    assert(rnn->num_layers < rnn->max_num_layers);
    rnn->layers[rnn->num_layers] = layer;
    rnn->num_layers = rnn->num_layers + 1;
}

void e_rnn_forward(E_RNN rnn, Matrix input){
    for(int i = 0; i < rnn.num_layers;i++){
        mat_dot(rnn.layers[i].state, rnn.layers[i].s_weights, rnn.layers[i].output);
        mat_dot(rnn.layers[i].output, rnn.layers[i].weights, input);
        mat_sum(rnn.layers[i].output, rnn.layers[i].output, rnn.layers[i].state);
        mat_sum(rnn.layers[i].output, rnn.layers[i].bias, rnn.layers[i].output);

        switch(rnn.layers[i].activation)
        {
            case SIGMOID:
                mat_sigmoid_f(rnn.layers[i].output, rnn.layers[i].output);
                break;
            case RELU:
                mat_relu_f(rnn.layers[i].output, rnn.layers[i].output);
                break;
            default:
                break;
        } 
        input = rnn.layers[i].output;
    }
}

Matrix e_rnn_cost(E_RNN rnn, Matrix td_input, Matrix td_output){
    Matrix distance = mat_alloc(td_output.rows,1);;
    Matrix error = mat_alloc(td_output.rows,1);
    Matrix mse = mat_alloc(td_output.rows,1);
    
    mat_fill(mse,0);

    Matrix x = mat_alloc(td_input.rows, 1);
    Matrix out = mat_alloc(td_output.rows,1);

    Matrix y;

    for(int col = 0; col < td_input.cols;col++){
        for(int row = 0; row < td_input.rows;row++){
            MAT_INDEX(x,row,0) = MAT_INDEX(td_input,row,col);
        }
        
        for(int row = 0; row < td_output.rows;row++){
            MAT_INDEX(out,row,0) = MAT_INDEX(td_output,row,col);
        }

        e_rnn_forward(rnn, x);
        
        mat_diff(distance,out,rnn.layers[rnn.num_layers - 1].output);
        mat_mult_elem(error,distance,distance);// I don't think this has to be matmul
        mat_sum(mse,mse,error);

    }
    mat_div(mse,mse,td_input.cols);

    return mse;
}

void e_rnn_train(E_RNN m, Matrix td_x, Matrix td_y, float lr, size_t epochs){
    for(int epoch = 0; epoch < epochs; epoch++){
        e_rnn_backprop(m,td_x,td_y,lr);
        // mat_print(e_rnn_cost(m, td_x, td_y));
    }
}

void e_rnn_backprop(E_RNN rnn, Matrix td_x, Matrix td_y, float lr){
    Matrix *error = malloc(rnn.num_layers * sizeof(Matrix));
    Matrix *dW = malloc(rnn.num_layers * sizeof(Matrix));
    Matrix *dWs = malloc(rnn.num_layers * sizeof(Matrix));
    Matrix *dB = malloc(rnn.num_layers * sizeof(Matrix));
    Matrix x = mat_alloc(td_x.rows, 1);
    Matrix y = mat_alloc(td_y.rows, 1);

    for(int layer = 0; layer < rnn.num_layers; layer++){
        dW[layer] = mat_alloc(rnn.layers[layer].weights.rows, rnn.layers[layer].weights.cols);
        dWs[layer] = mat_alloc(rnn.layers[layer].s_weights.rows, rnn.layers[layer].s_weights.cols);
        dB[layer] = mat_alloc(rnn.layers[layer].bias.rows, rnn.layers[layer].bias.cols);

        mat_fill(dW[layer],0);
        mat_fill(dWs[layer],0);
        mat_fill(dB[layer],0);
    }

    for(int t_col = 0; t_col < td_x.cols; t_col++){
        for(int x_index = 0; x_index < td_x.rows;x_index++){
            MAT_INDEX(x,x_index,0) = MAT_INDEX(td_x,x_index,t_col);
        }
        for(int y_index = 0; y_index < td_y.rows;y_index++){
            MAT_INDEX(y,y_index,0) = MAT_INDEX(td_y,y_index,t_col);
        }

        e_rnn_forward(rnn, x);

        for(int layer = rnn.num_layers - 1; layer >= 0; layer--){
            if(layer == rnn.num_layers - 1){
                Matrix tmp = mat_alloc(rnn.layers[layer].output.rows, rnn.layers[layer].output.cols);
                error[layer] = mat_alloc(y.rows, y.cols);
                mat_diff(error[layer], y, rnn.layers[layer].output);
                mat_mult(error[layer], error[layer], -2);

                switch(rnn.layers[layer].activation)
                {
                    case SIGMOID:
                        mat_sigmoid_f_deriv(tmp, rnn.layers[layer].output);
                        break;
                    case RELU:
                        mat_relu_f_deriv(tmp, rnn.layers[layer].output);
                        break;
                    default:
                        break;
                }

                mat_mult_elem(error[layer], error[layer], tmp);
            }
            else{
                Matrix tmp0 = mat_alloc(rnn.layers[layer].output.rows, rnn.layers[layer].output.cols);//activation
                Matrix tmp1 = mat_alloc(rnn.layers[layer].output.rows, rnn.layers[layer].output.cols);//pre-activation error

                error[layer] = mat_alloc(rnn.layers[layer].output.rows, rnn.layers[layer].output.cols);
                mat_dot(tmp1, mat_transpose(rnn.layers[layer+1].weights), error[layer+1]);
            
                switch (rnn.layers[layer].activation)
                {
                    case SIGMOID:
                        mat_sigmoid_f_deriv(tmp0, rnn.layers[layer].output);
                        break;
                    case RELU:
                        mat_relu_f_deriv(tmp0, rnn.layers[layer].output);
                        break;
                    default:
                        break;
                }
                mat_mult_elem(error[layer], tmp1, tmp0);

                mat_free(tmp0);
                mat_free(tmp1);
            }
        }

        for(int layer = 0; layer < rnn.num_layers; layer++){

            if(layer == 0){
                Matrix tmp = mat_alloc(dW[layer].rows, dW[layer].cols);
                Matrix tmp1 = mat_alloc(dWs[layer].rows, dWs[layer].cols);
                mat_dot(tmp, error[layer], mat_transpose(x));
                mat_dot(tmp1, error[layer], mat_transpose(rnn.layers[layer].state));
                mat_sum(dW[layer],dW[layer], tmp);
                mat_sum(dWs[layer],dWs[layer], tmp1);
                mat_sum(dB[layer],dB[layer], error[layer]);
                
                mat_free(tmp);
                mat_free(tmp1);
            }

            else{
                Matrix tmp = mat_alloc(dW[layer].rows, dW[layer].cols);
                Matrix tmp1 = mat_alloc(dWs[layer].rows, dWs[layer].cols);
                mat_dot(tmp, error[layer], mat_transpose(rnn.layers[layer-1].output));
                mat_dot(tmp1, error[layer], mat_transpose(rnn.layers[layer].state));
                mat_sum(dW[layer],dW[layer], tmp);
                mat_sum(dWs[layer],dWs[layer], tmp1);
                mat_sum(dB[layer],dB[layer], error[layer]);
                mat_free(tmp);
                mat_free(tmp1);
            }
        }
    }
    
    for(int layer = 0; layer < rnn.num_layers;layer++){
        mat_mult(dW[layer], dW[layer], (lr/td_x.cols));
        mat_mult(dWs[layer], dWs[layer], (lr/td_x.cols));

        mat_diff(rnn.layers[layer].weights, rnn.layers[layer].weights, dW[layer]);
        mat_diff(rnn.layers[layer].s_weights, rnn.layers[layer].s_weights, dWs[layer]);
        mat_diff(rnn.layers[layer].bias, rnn.layers[layer].bias, dB[layer]); 

        mat_free(dW[layer]);
        mat_free(dWs[layer]);
        mat_free(dB[layer]);
        mat_free(error[layer]);
    }

    free(dW);
    free(dWs);
    free(dB);
    free(error);

    mat_free(x);
    mat_free(y);
}

E_RNN e_rnn_copy(E_RNN rnn){
    return rnn;
}

void e_rnn_layer_free(E_RNN_Layer layer){
    mat_free(layer.weights);
    mat_free(layer.bias);
    mat_free(layer.output);
    mat_free(layer.state);
    mat_free(layer.s_weights);
}

void e_rnn_free(E_RNN rnn){
    for(size_t i = 0; i < rnn.num_layers; i++){
        e_rnn_layer_free(rnn.layers[i]);
    }
    free(rnn.layers);
}
#endif // RNN_H_


#ifdef RNN_IMPLEMENTATION

#endif // NN_IMPLEMENTATION