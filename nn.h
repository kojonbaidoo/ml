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

float rand_float();
float sigmoid_f(float value);
float sigmoid_f_deriv(float value);

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

void mat_sigmoid_f(Matrix mat1, Matrix mat0){
    assert(mat0.rows == mat1.rows);
    assert(mat0.cols == mat1.cols);

    for(int row = 0; row < mat0.rows;row++){
        for(int col = 0; col < mat0.cols; col++){
            MAT_INDEX(mat1,row,col) = sigmoid_f(MAT_INDEX(mat0,row,col));
        }
    }
}

void mat_free(Matrix mat){
    free(mat.vals);
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