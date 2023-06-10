#define NN_IMPLEMENTATION
#include "nn.h"
#include <assert.h>

#define MAT_INDEX(MAT,ROW,COL) MAT.vals[(ROW) * (MAT.cols) + (COL)]

void mat_dot_test();
void mat_sum_test();
void mat_diff_test();
void mat_div_test();
void mat_alloc_test();
void mat_fill_test();

void sigmoid_f_test();
void mat_sigmoid_f_test();

int main(void){
    mat_alloc_test();
    mat_dot_test();
    mat_sum_test();
    mat_diff_test();
    mat_div_test();
    mat_fill_test();
    sigmoid_f_test();
    mat_sigmoid_f_test();
    return 0;
}

void mat_alloc_test(){
    Matrix mat0 = mat_alloc(2,1);
    assert(mat0.rows == 2);
    assert(mat0.cols == 1);

    Matrix mat1 = mat_alloc(1,2);
    assert(mat1.rows == 1);
    assert(mat1.cols == 2);

    Matrix mat2 = mat_alloc(20,11);
    assert(mat2.rows == 20);
    assert(mat2.cols == 11);

    printf("Tests Passed - mat_alloc: Random Matrix\n");
}

void mat_dot_test(){
    Matrix mat0 = mat_alloc(2,2);
    mat0.vals[0] = 1.0;
    mat0.vals[1] = 2.0;
    mat0.vals[2] = 3.0;
    mat0.vals[3] = 4.0;

    Matrix mat1 = mat_alloc(2,2);
    mat1.vals[0] = 1.0;
    mat1.vals[1] = 0.0;
    mat1.vals[2] = 0.0;
    mat1.vals[3] = 1.0;

    Matrix mat2;
    mat2.rows = mat0.rows;
    mat2.cols = mat1.cols;
    mat2.vals = malloc(mat2.rows * mat2.cols * sizeof(float));
    
    mat_dot(mat2, mat0, mat1);
    
    for(int row = 0; row < mat2.rows;row++){
        for(int col = 0; col < mat2.cols; col++){
            assert(MAT_INDEX(mat0,row,col) == MAT_INDEX(mat2, row,col));
        }
    }
    printf("Tests Passed - mat_dot: Identity Matrix\n");
    free(mat0.vals);
    free(mat1.vals);
    free(mat2.vals);
}

void mat_sum_test(){
    Matrix mat0 = mat_alloc(2,2);
    mat0.vals[0] = 1.0;
    mat0.vals[1] = 2.0;
    mat0.vals[2] = 3.0;
    mat0.vals[3] = 4.0;

    Matrix mat2;
    mat2.rows = mat0.rows;
    mat2.cols = mat0.cols;
    mat2.vals = malloc(mat0.rows * mat0.cols * sizeof(float));
    
    mat_sum(mat2, mat0, mat0);
    
    for(int row = 0; row < mat2.rows;row++){
        for(int col = 0; col < mat2.cols; col++){
            assert((2 * MAT_INDEX(mat0, row, col)) == MAT_INDEX(mat2, row,col));
        }
    }
    printf("Tests Passed - mat_sum: Same Matrix\n");
}

void mat_diff_test(){
    Matrix mat0 = mat_alloc(2,2);
    mat0.vals[0] = 1.0;
    mat0.vals[1] = 2.0;
    mat0.vals[2] = 3.0;
    mat0.vals[3] = 4.0;

    Matrix mat2;
    mat2.rows = mat0.rows;
    mat2.cols = mat0.cols;
    mat2.vals = malloc(mat0.rows * mat0.cols * sizeof(float));
    
    mat_diff(mat2, mat0, mat0);
    
    for(int row = 0; row < mat2.rows;row++){
        for(int col = 0; col < mat2.cols; col++){
            assert(0 == MAT_INDEX(mat2, row,col));
        }
    }
    printf("Tests Passed - mat_diff: Same Matrix\n");
}

void mat_div_test(){
    Matrix mat0 = mat_alloc(2,2);
    mat0.vals[0] = 1.0;
    mat0.vals[1] = 2.0;
    mat0.vals[2] = 3.0;
    mat0.vals[3] = 4.0;

    Matrix mat2;
    mat2.rows = mat0.rows;
    mat2.cols = mat0.cols;
    mat2.vals = malloc(mat0.rows * mat0.cols * sizeof(float));
    
    mat_div(mat2, mat0, 2);
    
    for(int row = 0; row < mat2.rows;row++){
        for(int col = 0; col < mat2.cols; col++){
            assert((MAT_INDEX(mat0,row,col)/2) == MAT_INDEX(mat2, row,col));
        }
    }
    printf("Tests Passed - mat_div: Matrix - (Matrix / 2)\n");
}

void mat_fill_test(){
    Matrix mat0 = mat_alloc(2,1);
    mat_fill(mat0, 1);
    for(int row = 0; row < mat0.rows;row++){
        for(int col = 0; col < mat0.cols; col++){
            assert((MAT_INDEX(mat0, row, col)) == 1);
        }
    }

    Matrix mat1 = mat_alloc(2,10);
    mat_fill(mat1, 3);
    for(int row = 0; row < mat1.rows;row++){
        for(int col = 0; col < mat1.cols; col++){
            assert((MAT_INDEX(mat1, row, col)) == 3);
        }
    }

    printf("Tests Passed - mat_fill: Random numbers\n");
}

void mat_sigmoid_f_test(){
    Matrix mat0 = mat_alloc(2,5);

    mat_fill(mat0, 0);
    mat_sigmoid_f(mat0,mat0);
    for(int row = 0; row < mat0.rows;row++){
        for(int col = 0; col < mat0.cols; col++){
            assert((MAT_INDEX(mat0, row, col)) == 0.5);
        }
    }

    Matrix mat1 = mat_alloc(1,1);
    mat_fill(mat1, 0);
    mat_sigmoid_f(mat1,mat1);
    for(int row = 0; row < mat1.rows;row++){
        for(int col = 0; col < mat1.cols; col++){
            assert((MAT_INDEX(mat1, row, col)) == 0.5);
        }
    }

    printf("Tests Passed - mat_sigmoid_f: 0 matrix\n");
}

void sigmoid_f_test(){
    assert(0.5 == sigmoid_f(0.0));
    printf("Tests Passed - sigmoid_f: 0.5\n");
}
