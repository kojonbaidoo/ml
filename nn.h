#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <math.h>

#define SIGMOID 0
#define RELU 1
#define PARAM_COUNT(PARAMS) (sizeof(PARAMS) / sizeof(PARAMS[0]))
#define MAT_INDEX(MAT,ROW,COL) MAT.vals[(ROW) * (MAT.cols) + (COL)]

typedef struct {
    size_t rows;
    size_t cols;
    float *vals;
} Matrix;

typedef struct {
    size_t layers;
    Matrix *w;
    Matrix *b;
    Matrix *a;
} Model;

float rand_float();
float sigmoid_f(float value);
float sigmoid_f_deriv(float value);
float cost(Model m, Matrix td_x, Matrix td_y);
void forward(Model m, Matrix input);
void backpropagation(Model m, Matrix td_x, Matrix td_y, float lr);

#ifndef NN_H_
Matrix mat_alloc(size_t rows, size_t cols);
void mat_dot(Matrix mat3, Matrix mat1, Matrix mat2);
void mat_sum(Matrix mat2, Matrix mat1, Matrix mat0);
void mat_diff(Matrix mat2, Matrix mat0, Matrix mat1);
void mat_div(Matrix mat2, Matrix mat0, float value);
void mat_mult(Matrix mat2, Matrix mat0, float value);
void mat_mult_elem(Matrix mat2, Matrix mat0, Matrix mat1);
void mat_print(Matrix mat);
void mat_rand(Matrix m);
void mat_fill(Matrix mat, float value);
void mat_sigmoid_f(Matrix m0, Matrix m1);
void mat_sigmoid_f_deriv(Matrix m0, Matrix m1);

void mat_free(Matrix mat);

Model model_alloc(int (*params)[3], size_t layers);
void model_train(Model m, Matrix td_x, Matrix td_y, float lr, size_t epochs);
void model_free(Model model);

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

Model model_alloc(int (*params)[3], size_t layers){
    Model m;
    m.layers = layers;

    m.w = malloc(m.layers * sizeof(Matrix));
    m.b = malloc(m.layers * sizeof(Matrix));
    m.a = malloc(m.layers * sizeof(Matrix));

    // params: {num_inputs, num_neurons, activation}
    for(int layer = 0;layer < layers;layer++){
        if(layer > 0){assert(params[layer][0] == params[layer - 1][1]);}// Inputs in the current layer should match neurons in the last layer

        m.w[layer] = mat_alloc(params[layer][1], params[layer][0]);
        m.b[layer] = mat_alloc(params[layer][1], 1);
        m.a[layer] = mat_alloc(params[layer][1], 1);
    }

    return m;
}

void model_print(Model model){
    for(int layer = 0;layer < model.layers; layer++){
        printf("-------------------- Layer: %d --------------------\n",layer);
        printf("w%d = ",layer);
        mat_print(model.w[layer]);

        printf("b%d = ",layer);
        mat_print(model.b[layer]);

        printf("a%d = ",layer);
        mat_print(model.a[layer]);
    }
}

void model_free(Model model){
    for(int layer = 0;layer < model.layers; layer++){
        mat_free(model.w[layer]);
        mat_free(model.b[layer]);
        mat_free(model.a[layer]);
    }
}

void model_train(Model m, Matrix td_x, Matrix td_y, float lr, size_t epochs){
    for(int epoch = 0; epoch < epochs; epoch++){
        backpropagation(m,td_x,td_y,1);
    }
}

void backpropagation(Model m, Matrix td_x, Matrix td_y, float lr){
    Matrix *error = malloc(m.layers * sizeof(Matrix));
    Matrix *dW = malloc(m.layers * sizeof(Matrix));
    Matrix *dB = malloc(m.layers * sizeof(Matrix));
    Matrix x = mat_alloc(td_x.rows, 1);
    Matrix y = mat_alloc(td_y.rows, 1);

    for(int layer = 0; layer < m.layers; layer++){
        dW[layer] = mat_alloc(m.w[layer].rows, m.w[layer].cols);
        dB[layer] = mat_alloc(m.b[layer].rows, m.b[layer].cols);
    }

    for(int t_col = 0; t_col < td_x.cols; t_col++){
        for(int x_index = 0; x_index < td_x.rows;x_index++){
            MAT_INDEX(x,x_index,0) = MAT_INDEX(td_x,x_index,t_col);
        }
        for(int y_index = 0; y_index < td_y.rows;y_index++){
            MAT_INDEX(y,y_index,0) = MAT_INDEX(td_y,y_index,t_col);
        }

        forward(m, x);

        for(int layer = m.layers - 1; layer >= 0; layer--){
            if(layer == m.layers - 1){
                Matrix tmp = mat_alloc(m.a[layer].rows, m.a[layer].cols);
                error[layer] = mat_alloc(y.rows, y.cols);
                mat_diff(error[layer], y, m.a[layer]);
                mat_mult(error[layer], error[layer], -2);
                mat_sigmoid_f_deriv(tmp, m.a[layer]);
                mat_mult_elem(error[layer], error[layer], tmp);
                
                mat_free(tmp);
            }
            else{
                Matrix tmp = mat_alloc(m.a[layer].rows, m.a[layer].cols);
                error[layer] = mat_alloc(error[layer+1].rows,  m.w[layer+1].cols);
                mat_dot(error[layer], error[layer+1], m.w[layer+1]);
                mat_sigmoid_f_deriv(tmp, m.a[layer]);
                mat_mult_elem(error[layer], error[layer], mat_transpose(tmp));

                mat_free(tmp);
            }
        }

        for(int layer = 0; layer < m.layers; layer++){

            if(layer == 0){
                Matrix tmp = mat_alloc(td_x.rows, error[layer].cols);
                mat_dot(tmp, x, error[layer]);
                mat_sum(dW[layer],dW[layer], mat_transpose(tmp));
                mat_sum(dB[layer],dB[layer], mat_transpose(error[layer]));
                mat_free(tmp);
            }

            else{
                Matrix tmp = mat_alloc(m.a[layer - 1].rows, error[layer].cols);
                mat_dot(tmp, m.a[layer - 1], error[layer]);
                mat_sum(dW[layer],dW[layer], mat_transpose(tmp));
                mat_sum(dB[layer],dB[layer], mat_transpose(error[layer]));
                mat_free(tmp);
            }

        }
    }
    
    for(int layer = 0; layer < m.layers;layer++){
        mat_mult(dW[layer], dW[layer], (lr/td_x.cols));
        mat_diff(m.w[layer], m.w[layer], dW[layer]);
        mat_diff(m.b[layer], m.b[layer], dB[layer]); 
    }

    free(dW);
    free(dB);
    free(error);

    mat_free(x);
    mat_free(y);
}

float cost(Model m, Matrix td_x, Matrix td_y){
    assert(td_x.cols == td_y.cols);

    size_t num_data = td_x.cols;
    Matrix x = mat_alloc(td_x.rows, 1);
    Matrix y = mat_alloc(td_y.rows, 1);

    float cost = 0.0;
    float d = 0;

    for(int t_col = 0; t_col < num_data; t_col++){
        for(int x_index = 0; x_index < td_x.rows;x_index++){
            MAT_INDEX(x,x_index,0) = MAT_INDEX(td_x,x_index,t_col);
        }
        for(int y_index = 0; y_index < td_y.rows;y_index++){
            MAT_INDEX(y,y_index,0) = MAT_INDEX(td_y,y_index,t_col);
        }

        forward(m,x);
        
        for(int out_index = 0; out_index < y.rows; out_index++){
            d = MAT_INDEX(m.a[m.layers - 1],out_index,0) - MAT_INDEX(y,out_index,0);
            cost += d * d;
        }
    }

    mat_free(x);
    mat_free(y);

    return cost / num_data;
}

void forward(Model m, Matrix input){
    assert(m.w[0].cols == input.rows);

    for(int i = 0; i < m.layers; i++){
        if(i == 0){
            mat_dot(m.a[i], m.w[i], input);}
        else{
            mat_dot(m.a[i], m.w[i], m.a[i-1]);}
        
        mat_sum(m.a[i], m.b[i], m.a[i]);
        mat_sigmoid_f(m.a[i], m.a[i]);
    }   
}


float sigmoid_f(float value){
    return 1.0 / (1.0 + expf(-value));
}

float sigmoid_f_deriv(float value){
    return value * (1 - value);
}

float rand_float(){
    return (float) rand() / (RAND_MAX);
}
#endif // NN_H_


#ifdef NN_IMPLEMENTATION

#endif // NN_IMPLEMENTATION