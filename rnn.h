#include <nn.h>

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
void e_rnn_add(E_RNN *rnn, Layer layer);
void e_rnn_forward(E_RNN rnn, Matrix input);
void e_rnn_train(E_RNN rnn, Matrix td_x, Matrix td_y, float lr, size_t epochs);
void e_rnn_backprop(E_RNN rnn, Matrix td_x, Matrix td_y, float lr);
E_RNN e_rnn_copy(E_RNN rnn);

#ifndef RNN_H_
void e_rnn_layer_free(Layer layer);
void e_rnn_free(MLP mlp);

#define RNN_H_

E_RNN e_rnn_alloc(size_t num_layers){
    E_RNN rnn;
    rnn.num_layers = 0;
    rnn.max_num_layers = num_layers;
    rnn.layers = malloc(num_layers * sizeof(E_RNN_Layer));

    return rnn;
}

void e_rnn_add(E_RNN *rnn, E_RNN_Layer layer){
    assert(rnn->num_layers < rnn->max_num_layers);
    rnn->layers[rnn->num_layers] = layer;
    rnn->num_layers = rnn->num_layers + 1;
}

void mlp_forward(E_RNN rnn, Matrix input){
    for(int i = 0; i < rnn.num_layers;i++){
        mat_dot(rnn.layers[i].output, rnn.layers[i].weights, input);
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

Matrix mlp_cost(MLP mlp, Matrix td_input, Matrix td_output){
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

        mlp_forward(mlp, x);
        
        mat_diff(distance,out,mlp.layers[mlp.num_layers - 1].output);
        mat_mult_elem(error,distance,distance);// I don't think this has to be matmul
        mat_sum(mse,mse,error);

    }
    mat_div(mse,mse,td_input.cols);

    return mse;
}

void mlp_train(MLP m, Matrix td_x, Matrix td_y, float lr, size_t epochs){
    for(int epoch = 0; epoch < epochs; epoch++){
        mlp_backprop(m,td_x,td_y,lr);
        // mat_print(mlp_cost(m, td_x, td_y));
    }
}

void mlp_backprop(MLP mlp, Matrix td_x, Matrix td_y, float lr){
    Matrix *error = malloc(mlp.num_layers * sizeof(Matrix));
    Matrix *dW = malloc(mlp.num_layers * sizeof(Matrix));
    Matrix *dB = malloc(mlp.num_layers * sizeof(Matrix));
    Matrix x = mat_alloc(td_x.rows, 1);
    Matrix y = mat_alloc(td_y.rows, 1);

    for(int layer = 0; layer < mlp.num_layers; layer++){
        dW[layer] = mat_alloc(mlp.layers[layer].weights.rows, mlp.layers[layer].weights.cols);
        dB[layer] = mat_alloc(mlp.layers[layer].bias.rows, mlp.layers[layer].bias.cols);

        mat_fill(dW[layer],0);
        mat_fill(dB[layer],0);
    }

    for(int t_col = 0; t_col < td_x.cols; t_col++){
        for(int x_index = 0; x_index < td_x.rows;x_index++){
            MAT_INDEX(x,x_index,0) = MAT_INDEX(td_x,x_index,t_col);
        }
        for(int y_index = 0; y_index < td_y.rows;y_index++){
            MAT_INDEX(y,y_index,0) = MAT_INDEX(td_y,y_index,t_col);
        }

        mlp_forward(mlp, x);

        for(int layer = mlp.num_layers - 1; layer >= 0; layer--){
            if(layer == mlp.num_layers - 1){
                Matrix tmp = mat_alloc(mlp.layers[layer].output.rows, mlp.layers[layer].output.cols);
                error[layer] = mat_alloc(y.rows, y.cols);
                mat_diff(error[layer], y, mlp.layers[layer].output);
                mat_mult(error[layer], error[layer], -2);

                switch(mlp.layers[layer].activation)
                {
                    case SIGMOID:
                        mat_sigmoid_f_deriv(tmp, mlp.layers[layer].output);
                        break;
                    case RELU:
                        mat_relu_f_deriv(tmp, mlp.layers[layer].output);
                        break;
                    default:
                        break;
                }

                mat_mult_elem(error[layer], error[layer], tmp);
                mat_free(tmp);
            }
            else{
                Matrix tmp0 = mat_alloc(mlp.layers[layer].output.rows, mlp.layers[layer].output.cols);
                Matrix tmp1 = mat_alloc(mlp.layers[layer].output.cols, mlp.layers[layer].output.rows);

                error[layer] = mat_alloc(mlp.layers[layer].output.rows, mlp.layers[layer].output.cols);

                mat_dot(tmp1, mat_transpose(error[layer+1]), mlp.layers[layer+1].weights);

                switch (mlp.layers[layer].activation)
                {
                    case SIGMOID:
                        mat_sigmoid_f_deriv(tmp0, mlp.layers[layer].output);
                        break;
                    case RELU:
                        mat_relu_f_deriv(tmp0, mlp.layers[layer].output);
                        break;
                    default:
                        break;
                }
                mat_mult_elem(error[layer], mat_transpose(tmp1), tmp0);

                mat_free(tmp0);
                mat_free(tmp1);
            }
        }

        for(int layer = 0; layer < mlp.num_layers; layer++){

            if(layer == 0){
                Matrix tmp = mat_alloc(dW[layer].rows, dW[layer].cols);
                mat_dot(tmp, error[layer], mat_transpose(x));
                mat_sum(dW[layer],dW[layer], tmp);
                mat_sum(dB[layer],dB[layer], error[layer]);
                
                mat_free(tmp);
            }

            else{
                Matrix tmp = mat_alloc(dW[layer].rows, dW[layer].cols);
                mat_dot(tmp, error[layer], mat_transpose(mlp.layers[layer-1].output));
                mat_sum(dW[layer],dW[layer], tmp);
                mat_sum(dB[layer],dB[layer], error[layer]);
                mat_free(tmp);
            }
        }
    }
    
    for(int layer = 0; layer < mlp.num_layers;layer++){
        mat_mult(dW[layer], dW[layer], (lr/td_x.cols));
        mat_diff(mlp.layers[layer].weights, mlp.layers[layer].weights, dW[layer]);
        mat_diff(mlp.layers[layer].bias, mlp.layers[layer].bias, dB[layer]); 

        mat_free(dW[layer]);
        mat_free(dB[layer]);
        mat_free(error[layer]);
    }

    free(dW);
    free(dB);
    free(error);

    mat_free(x);
    mat_free(y);
}

E_RNN e_rnn_copy(E_RNN rnn){
    return rnn;
}

void e_rnn_layer_free(Layer layer){
    mat_free(layer.weights);
    mat_free(layer.bias);
    mat_free(layer.output);
    mat_free(layer.state);
    mat_free(layer.s_weight);
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