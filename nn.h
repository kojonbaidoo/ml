#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <math.h>

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
mlp_forward(MLP *mlp, Matrix input);
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

#define NN_H_

#define MAT_INDEX(MAT,ROW,COL) MAT.vals[(ROW) * (MAT.cols) + (COL)]


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

void mat_free(Matrix mat){
    free(mat.vals);
}

Layer layer_alloc(size_t num_inputs, size_t num_neurons, int activation){
    Layer layer;
    layer.neurons = num_neurons;
    layer.weights = mat_alloc(num_neurons, num_inputs);
    layer.bias = mat_alloc(num_neurons, 1);
    layer.output = mat_alloc(num_neurons, 1);
    layer.activation = activation;
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

void mlp_forward(MLP *mlp, Matrix input){
    for(int i = 0; i < mlp->num_layers;i++){
        mat_dot(mlp->layers[i].output, mlp->layers[i].weights, input);
        mat_sum(mlp->layers[i].output, mlp->layers[i].bias, mlp->layers[i].output);
        if(mlp->layers[i].activation == SIGMOID){
            mat_sigmoid_f(mlp->layers[i].output, mlp->layers[i].output);
        }
        input = mlp->layers[i].output;
    }
}

Matrix mlp_cost(MLP mlp, Matrix td_input, Matrix td_output){
    Matrix distance = mat_alloc(td_output.rows,1);;
    Matrix error = mat_alloc(td_output.rows,1);
    Matrix mse = mat_alloc(td_output.rows,1);

    Matrix x = mat_alloc(td_input.rows, 1);
    Matrix out = mat_alloc(td_output.rows,1);

    Matrix y;

    for(int col = 0; col < td_input.cols;col++){
        for(int row = 0; row < td_input.rows;row++){
            MAT_INDEX(x,row,1) = MAT_INDEX(td_input,row,col);
        }
        
        for(int row = 0; row < td_output.rows;row++){
            MAT_INDEX(out,row,1) = MAT_INDEX(td_output,row,col);
        }

        mlp_forward(&mlp, x);
        
        mat_diff(distance,out,mlp.layers[mlp.num_layers - 1].output);
        mat_dot(error,distance,distance);// I don't think this has to be matmul
        mat_sum(mse,mse,error);
    }

    mat_div(mse,mse,td_input.cols);

    return mse;
}

void mlp_backprop(MLP *mlp, Matrix td_x, Matrix td_y, float lr){

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
#endif // NN_H_


#ifdef NN_IMPLEMENTATION

#endif // NN_IMPLEMENTATION