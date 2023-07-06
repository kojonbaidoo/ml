#define NN_IMPLEMENTATION
#include "nn.h"
#include <assert.h>

// #define MAT_INDEX(MAT,ROW,COL) MAT.vals[(ROW) * (MAT.cols) + (COL)]

void mat_dot_test();
void mat_sum_test();
void mat_copy_test();
void mat_diff_test();
void mat_div_test();
void mat_alloc_test();
void mat_fill_test();

void layer_alloc_test();

void mlp_alloc_test();
void mlp_add_test();
void mlp_forward_test();

void sigmoid_f_test();
void mat_sigmoid_f_test();
void relu_f_test();
void mat_relu_f_test();

void rand_float_range_test();

void mat_dot_multithreaded_test();

int main(void){
    rand_float_range_test();
    mat_alloc_test();
    mat_dot_test();
    mat_dot_multithreaded_test();
    mat_sum_test();
    mat_copy_test();
    mat_diff_test();
    mat_div_test();
    mat_fill_test();
    sigmoid_f_test();
    mat_sigmoid_f_test();
    relu_f_test();
    mat_relu_f_test();
    layer_alloc_test();
    mlp_alloc_test();
    mlp_add_test();
    mlp_forward_test();

    return 0;
}

void rand_float_range_test(){
    float t;
    float epsilon = 0.0000001;
    t = rand_float_range(-0.2,0.2);
    assert(t <= (0.2 + epsilon) && t >= (-0.2 - epsilon));

    t = rand_float_range(0,0.5);
    assert(t <= (0.5 + epsilon) && t >= (0 - epsilon));

    t = rand_float_range(-0.2,0);
    assert(t <= (0.0 + epsilon)  && t >= (-0.2 - epsilon));

    printf("Tests Passed - rand_float_range: Different ranges\n");
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

void mat_dot_multithreaded_test(){
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
    
    mat_dot_multithreaded(mat2, mat0, mat1);

    for(int row = 0; row < mat2.rows;row++){
        for(int col = 0; col < mat2.cols; col++){
            assert(MAT_INDEX(mat0,row,col) == MAT_INDEX(mat2, row,col));
        }
    }
    printf("Tests Passed - mat_dot_multithreaded: Identity Matrix\n");
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

void mat_copy_test(){
    Matrix mat0 = mat_alloc(2,2);
    Matrix mat2 = mat_alloc(2,2);
    mat_fill(mat0,1);
    mat_fill(mat2,0);

    mat_copy(mat2, mat0);
    
    for(int row = 0; row < mat2.rows;row++){
        for(int col = 0; col < mat2.cols; col++){
            assert(1 == MAT_INDEX(mat2, row,col));
        }
    }
    printf("Tests Passed - mat_copy: Matrix\n");
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

void mat_relu_f_test(){
    Matrix mat0 = mat_alloc(2,5);
    Matrix mat1 = mat_alloc(2,5);
    Matrix mat2 = mat_alloc(2,5);

    mat_fill(mat0,-2);
    mat_fill(mat1, 0);
    mat_fill(mat2, 2);

    mat_relu_f(mat0,mat0);
    mat_relu_f(mat1,mat1);
    mat_relu_f(mat2,mat2);

    for(int row = 0; row < mat0.rows;row++){
        for(int col = 0; col < mat0.cols; col++){
            assert((MAT_INDEX(mat0, row, col)) == 0);
            assert((MAT_INDEX(mat1, row, col)) == 0);
            assert((MAT_INDEX(mat2, row, col)) == 2);
        }
    }
    printf("Tests Passed - mat_relu_f: -2,0,2 matrix\n");
}

void relu_f_test(){
    assert(0 == relu_f(-0.5));
    assert(0 == relu_f(0.0));
    assert(0.5 == relu_f(0.5));

    printf("Tests Passed - relu_f: -0.5,0,0.5\n");
}

void layer_alloc_test(){
    Layer layer0 = layer_alloc(2,2,SIGMOID);
    assert(layer0.weights.rows == 2);
    assert(layer0.weights.cols == 2);
    assert(layer0.bias.rows == 2);
    assert(layer0.bias.cols == 1);
    assert(layer0.output.rows == 2);
    assert(layer0.output.cols == 1);
    assert(layer0.neurons == 2);
    assert(layer0.activation == SIGMOID);

    Layer layer1 = layer_alloc(10,23,SIGMOID);
    assert(layer1.weights.rows == 23);
    assert(layer1.weights.cols == 10);
    assert(layer1.bias.rows == 23);
    assert(layer1.bias.cols == 1);
    assert(layer1.output.rows == 23);
    assert(layer1.output.cols == 1);
    assert(layer1.neurons == 23);
    assert(layer1.activation == SIGMOID);

    printf("Tests Passed - Layer: 2\n");
}

void mlp_alloc_test(){
    MLP mlp0 = mlp_alloc(10);
    assert(mlp0.num_layers == 0);
    assert(mlp0.max_num_layers == 10);

    printf("Tests Passed - mlp_alloc\n");
}

void mlp_add_test(){
    Layer layer0 = layer_alloc(2,2,SIGMOID);
    Layer layer1 = layer_alloc(10,23,SIGMOID);

    MLP mlp = mlp_alloc(2);
    mlp_add(&mlp,layer0);
    assert(mlp.num_layers == 1);

    mlp_add(&mlp,layer1);
    assert(mlp.num_layers == 2);

    assert(layer0.weights.rows == mlp.layers[0].weights.rows);
    assert(layer0.weights.cols == mlp.layers[0].weights.cols);
    assert(layer0.bias.rows == mlp.layers[0].bias.rows);
    assert(layer0.bias.cols == mlp.layers[0].bias.cols);
    assert(layer0.output.rows == mlp.layers[0].output.rows);
    assert(layer0.output.cols == mlp.layers[0].output.cols);
    assert(layer0.neurons == mlp.layers[0].neurons);
    assert(layer0.activation == mlp.layers[0].activation);

    assert(layer1.weights.rows == mlp.layers[1].weights.rows);
    assert(layer1.weights.cols == mlp.layers[1].weights.cols);
    assert(layer1.bias.rows == mlp.layers[1].bias.rows);
    assert(layer1.bias.cols == mlp.layers[1].bias.cols);
    assert(layer1.output.rows == mlp.layers[1].output.rows);
    assert(layer1.output.cols == mlp.layers[1].output.cols);
    assert(layer1.neurons == mlp.layers[1].neurons);
    assert(layer1.activation == mlp.layers[1].activation);

    printf("Tests Passed - MLP creation\n");
}

void mlp_forward_test(){
    Layer layer0 = layer_alloc(2,2,SIGMOID);
    Layer layer1 = layer_alloc(2,1,SIGMOID);

    mat_fill(layer0.weights, 0);
    mat_fill(layer0.bias, 0);
    mat_fill(layer1.weights, 0);
    mat_fill(layer1.bias, 0);

    Matrix input = mat_alloc(2,1);
    mat_rand(input);

    MLP mlp = mlp_alloc(2);
    mlp_add(&mlp,layer0);
    mlp_add(&mlp,layer1);

    mlp_forward(mlp, input);
    
    assert(MAT_INDEX(mlp.layers[mlp.num_layers - 1].output,0,0) == 0.5);

    printf("Tests Passed - mlp_forward\n");
}

