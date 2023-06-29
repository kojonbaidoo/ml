#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <string.h>

#define SIGMOID 0
#define RELU 1

typedef struct {
    size_t rows;
    size_t cols;
    float *vals;
} Matrix;

typedef struct
{
    size_t neurons;
    Matrix weights;
    Matrix bias;
    Matrix output;
    int activation;
} Layer;

typedef struct{
    Layer *layers;
    size_t num_layers;
    size_t max_num_layers;
} MLP;

float rand_float();
float sigmoid_f(float value);
float sigmoid_f_deriv(float value);

Layer layer_alloc(size_t num_inputs, size_t num_neurons, int activation);

MLP mlp_alloc(size_t num_layers);
void mlp_add(MLP *mlp, Layer layer);
void mlp_forward(MLP mlp, Matrix input);
void mlp_train(MLP m, Matrix td_x, Matrix td_y, float lr, size_t epochs);
void mlp_backprop(MLP mlp, Matrix td_x, Matrix td_y, float lr);
MLP mlp_copy(MLP mlp);

#ifndef NN_H_
Matrix mat_alloc(size_t rows, size_t cols);
void mat_dot(Matrix mat3, Matrix mat1, Matrix mat2);
void mat_sum(Matrix mat2, Matrix mat1, Matrix mat0);
void mat_diff(Matrix mat2, Matrix mat0, Matrix mat1);
void mat_div(Matrix mat2, Matrix mat0, float value);
void mat_print(Matrix mat);
void mat_rand(Matrix m);
void mat_fill(Matrix mat, float value);
void mat_sigmoid_f(Matrix m0, Matrix m1);

void mat_free(Matrix mat);
void layer_free(Layer layer);
void mlp_free(MLP mlp);

void save_neural_network(const char* filename, MLP* net);

#define NN_H_

#define MAT_INDEX(MAT,ROW,COL) MAT.vals[(ROW) * (MAT.cols) + (COL)]


void save_neural_network(const char* filename, MLP* net) {
    char magic[8] = "NNETV1.0";
    FILE* file = fopen(filename, "wb");
    if (file == NULL) {
        printf("Could not open file for writing: %s\n", filename);
        return;
    }

    fwrite(magic, sizeof(magic), 1, file);
    fwrite(&net->num_layers, sizeof(net->num_layers), 1, file);

    for (int i = 0; i < net->num_layers; ++i) {
        Layer* layer = &net->layers[i];
        fwrite(&layer->weights.cols, sizeof(layer->weights.cols), 1, file);
        fwrite(&layer->neurons, sizeof(layer->neurons), 1, file);
        fwrite(&layer->activation, sizeof(layer->activation), 1, file);
        fwrite(layer->weights.vals, sizeof(float), layer->weights.cols * layer->neurons, file);
        fwrite(layer->bias.vals, sizeof(float), layer->neurons, file);
        
        // printf("%ld\n",layer->neurons);
    }

    fclose(file);
}

MLP* load_neural_network(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        printf("Could not open file for reading: %s\n", filename);
        return NULL;
    }

    char magic[8];
    fread(magic, sizeof(magic), 1, file);
    if (strcmp(magic, "NNETV1.0") != 0) {
        printf("Invalid magic string: %s\n", magic);
        return NULL;
    }

    size_t num_layers;
    size_t num_inputs;
    size_t num_neurons;
    int activation;

    fread(&num_layers, sizeof(num_layers), 1, file);
    MLP* net = malloc(sizeof(MLP));
    net->layers = malloc(sizeof(Layer) * num_layers);
    net->num_layers = num_layers;
    net->max_num_layers = num_layers;

    for (int i = 0; i < net->num_layers; ++i) {
        Layer* layer = &net->layers[i];
        fread(&num_inputs, sizeof(num_inputs), 1, file);
        fread(&num_neurons, sizeof(num_neurons), 1, file);
        fread(&activation, sizeof(activation), 1, file);

        layer->weights = mat_alloc(num_neurons, num_inputs);
        layer->bias = mat_alloc(num_neurons, 1);
        layer->output = mat_alloc(num_neurons, 1);
        layer->activation = activation;
        layer->neurons = num_neurons;

        fread(layer->weights.vals, sizeof(float), layer->weights.cols * layer->neurons, file);
        fread(layer->bias.vals, sizeof(float), layer->neurons, file);
    }

    fclose(file);

    return net;
}

Matrix mat_alloc(size_t rows, size_t cols){
    float *vals = malloc(rows * cols * sizeof(float));
    Matrix mat;
    mat.rows = rows;
    mat.cols = cols;
    mat.vals = vals;
    return mat;
}

void mat_dot(Matrix mat3, Matrix mat1, Matrix mat2){
    assert(mat1.cols == mat2.rows);
    assert(mat3.rows == mat1.rows);
    assert(mat3.cols == mat2.cols);

    for(int row = 0; row < mat3.rows;row++){
        for(int col = 0;col < mat3.cols;col++){
            mat3.vals[(row * mat3.cols) + col] = 0; 
            for(int m1_col = 0; m1_col < mat1.cols;m1_col++){
                MAT_INDEX(mat3,row,col) = MAT_INDEX(mat3,row,col) + MAT_INDEX(mat1,row,m1_col) * MAT_INDEX(mat2,m1_col,col);
            }
        }
    }
}

void mat_sum(Matrix mat2, Matrix mat1, Matrix mat0 ){
    assert(mat0.rows == mat1.rows);
    assert(mat0.cols == mat1.cols);
    assert(mat0.rows == mat2.rows);
    assert(mat0.cols == mat2.cols);

    for(int row = 0; row < mat2.rows;row++){
        for(int col = 0;col < mat2.cols;col++){
            MAT_INDEX(mat2,row,col) = MAT_INDEX(mat0,row,col) + MAT_INDEX(mat1,row,col);
        }
    }
}

void mat_diff(Matrix mat2, Matrix mat0, Matrix mat1){
    assert(mat0.rows == mat1.rows);
    assert(mat0.cols == mat1.cols);
    assert(mat0.rows == mat2.rows);
    assert(mat0.cols == mat2.cols);

    for(int row = 0; row < mat2.rows;row++){
        for(int col = 0;col < mat2.cols;col++){
            MAT_INDEX(mat2,row,col) = MAT_INDEX(mat0,row,col) - MAT_INDEX(mat1,row,col);
        }
    }
}

void mat_mult(Matrix mat2, Matrix mat0, float value){
    mat_div(mat2, mat0, (1/value));
}

void mat_mult_elem(Matrix mat2, Matrix mat0, Matrix mat1){
    assert(mat0.rows == mat1.rows);
    assert(mat0.cols == mat1.cols);

    for(int row = 0; row < mat0.rows; row++){
        for(int col = 0; col < mat0.cols; col++){
            MAT_INDEX(mat2, row, col) = MAT_INDEX(mat0, row, col) * MAT_INDEX(mat1, row, col);
        }
    }

}
void mat_div(Matrix mat2, Matrix mat0, float value){
    assert(mat0.rows == mat2.rows);
    assert(mat0.cols == mat2.cols);

    for(int row = 0; row < mat2.rows;row++){
        for(int col = 0;col < mat2.cols;col++){
            MAT_INDEX(mat2,row,col) = MAT_INDEX(mat0,row,col)/value;
        }
    }
}

void mat_print(Matrix mat){
    printf("[\n");
    for(int row = 0; row < mat.rows;row++){
        printf("\t");
        for(int col = 0; col < mat.cols; col++){
            printf("%f ", mat.vals[(row * mat.cols) + col]);
        }
        printf("\n");
    }
    printf("]\n\n");
}

void mat_rand(Matrix mat){
    srand(time(0));
    for(int row = 0; row < mat.rows;row++){
        for(int col = 0; col < mat.cols; col++){
            MAT_INDEX(mat,row,col) = rand_float();
        }
    }
}

void mat_rand_range(Matrix mat, float lower, float upper){
    srand(time(0));
    for(int row = 0; row < mat.rows;row++){
        for(int col = 0; col < mat.cols; col++){
            MAT_INDEX(mat,row,col) = rand_float();
        }
    }
}

void mat_fill(Matrix mat, float value){
    for(int row = 0; row < mat.rows; row++){
        for(int col = 0; col < mat.cols; col++){
            MAT_INDEX(mat,row,col) = value;
        }
    }
}

Matrix mat_transpose(Matrix mat){
    Matrix new_mat = mat_alloc(mat.cols, mat.rows);

    for(int row = 0; row < mat.rows; row++){
        for(int col = 0; col < mat.cols; col++){
            MAT_INDEX(new_mat, col, row) = MAT_INDEX(mat,row,col);
        }
    }

    return new_mat;
}

void mat_sigmoid_f(Matrix mat1, Matrix mat0){
    assert(mat0.rows == mat1.rows);
    assert(mat0.cols == mat1.cols);

    for(int row = 0; row < mat0.rows;row++){
        for(int col = 0; col < mat0.cols; col++){
            MAT_INDEX(mat1,row,col) = sigmoid_f(MAT_INDEX(mat0,row,col));
        }
    }
}

void mat_sigmoid_f_deriv(Matrix mat1, Matrix mat0){
    assert(mat0.rows == mat1.rows);
    assert(mat0.cols == mat1.cols);

    for(int row = 0; row < mat0.rows;row++){
        for(int col = 0; col < mat0.cols; col++){
            MAT_INDEX(mat1,row,col) = sigmoid_f_deriv(MAT_INDEX(mat0,row,col));
        }
    }
}

Layer layer_alloc(size_t num_inputs, size_t num_neurons, int activation){
    Layer layer;
    layer.neurons = num_neurons;
    layer.weights = mat_alloc(num_neurons, num_inputs);
    layer.bias = mat_alloc(num_neurons, 1);
    layer.output = mat_alloc(num_neurons, 1);
    layer.activation = activation;

    mat_rand(layer.weights);
    mat_rand(layer.bias);
    mat_fill(layer.output, 0);
    return layer;
}

MLP mlp_alloc(size_t num_layers){
    MLP mlp;
    mlp.num_layers = 0;
    mlp.max_num_layers = num_layers;
    mlp.layers = malloc(num_layers * sizeof(Layer));

    return mlp;
}

void mlp_add(MLP *mlp, Layer layer){
    assert(mlp->num_layers < mlp->max_num_layers);
    mlp->layers[mlp->num_layers] = layer;
    mlp->num_layers = mlp->num_layers + 1;
}

void mlp_forward(MLP mlp, Matrix input){
    for(int i = 0; i < mlp.num_layers;i++){
        mat_dot(mlp.layers[i].output, mlp.layers[i].weights, input);
        mat_sum(mlp.layers[i].output, mlp.layers[i].bias, mlp.layers[i].output);
        if(mlp.layers[i].activation == SIGMOID){
            mat_sigmoid_f(mlp.layers[i].output, mlp.layers[i].output);
        }
        input = mlp.layers[i].output;
    }
}

Matrix mlp_cost(MLP mlp, Matrix td_input, Matrix td_output){
    Matrix distance = mat_alloc(td_output.rows,1);;
    Matrix error = mat_alloc(td_output.rows,1);
    Matrix mse = mat_alloc(td_output.rows,1);
    
    mat_fill(mse,0);
    // mat_fill(distance,0);
    // mat_fill(error,0);

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
                mat_sigmoid_f_deriv(tmp, mlp.layers[layer].output);
                mat_mult_elem(error[layer], error[layer], tmp);
                
                mat_free(tmp);
            }
            else{
                Matrix tmp0 = mat_alloc(mlp.layers[layer].output.rows, mlp.layers[layer].output.cols);
                Matrix tmp1 = mat_alloc(mlp.layers[layer].output.cols, mlp.layers[layer].output.rows);

                // error[layer] = mat_alloc(error[layer+1].rows,  mlp.layers[layer+1].weights.cols);
                // mat_dot(error[layer], error[layer+1], mlp.layers[layer+1].weights);

                error[layer] = mat_alloc(mlp.layers[layer].output.rows, mlp.layers[layer].output.cols);

                mat_dot(tmp1, mat_transpose(error[layer+1]), mlp.layers[layer+1].weights);

                mat_sigmoid_f_deriv(tmp0, mlp.layers[layer].output);
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

MLP mlp_copy(MLP mlp){
    return mlp;
}

void mat_free(Matrix mat){
    free(mat.vals);
}

void layer_free(Layer layer){
    mat_free(layer.weights);
    mat_free(layer.bias);
    mat_free(layer.output);
}

void mlp_free(MLP mlp){
    for(size_t i = 0; i < mlp.num_layers; i++){
        layer_free(mlp.layers[i]);
    }
    free(mlp.layers);
}

float sigmoid_f(float value){
    return 1.0 / (1.0 + expf(-value));
}

float sigmoid_f_deriv(float value){
    return sigmoid_f(value) * (1 - sigmoid_f(value));
}

float rand_float(){
    return (float) rand() / (RAND_MAX);
}

float rand_float_range(float lower, float upper){
    return (float) ((rand() / (RAND_MAX)) * (upper + lower)) - lower ;
}
#endif // NN_H_


#ifdef NN_IMPLEMENTATION

#endif // NN_IMPLEMENTATION