#define NN_IMPLEMENTATION
#include "nn.h"
#include <assert.h>

#define MAT_INDEX(MAT,ROW,COL) MAT.vals[(ROW) * (MAT.cols) + (COL)]
#define EPOCHS 1
#define TRAINING_SAMPLE_SIZE 4

float training_data[TRAINING_SAMPLE_SIZE][3] = {
    {0,0,0},
    {0,1,0},
    {1,0,0},
    {1,1,1},
};

int main(void){
    Matrix td_x = mat_alloc(2, TRAINING_SAMPLE_SIZE);
    Matrix td_y = mat_alloc(1, TRAINING_SAMPLE_SIZE);
    int i;
    int j;

    for(i = 0;i < TRAINING_SAMPLE_SIZE;i++){
        for(j = 0; j < 2;j++){
            MAT_INDEX(td_x,j,i) = training_data[i][j];
        }
        MAT_INDEX(td_y,0,i) = training_data[i][j];
    }

    // Model creation
    Model xor;
    xor.layers = 2;

    xor.w = malloc(xor.layers * sizeof(Matrix));
    xor.b = malloc(xor.layers * sizeof(Matrix));
    xor.a = malloc(xor.layers * sizeof(Matrix));

    xor.w[0] = mat_alloc(2,2);
    xor.b[0] = mat_alloc(2,1);
    xor.a[0] = mat_alloc(2,1);

    xor.w[1] = mat_alloc(1,2);
    xor.b[1] = mat_alloc(1,1);
    xor.a[1] = mat_alloc(1,1);

    for(int layer = 0; layer < xor.layers; layer++){
        mat_rand(xor.w[layer]);
        mat_rand(xor.b[layer]);
        mat_rand(xor.a[layer]);
    }

    Matrix input = mat_alloc(td_x.rows, 1);
    MAT_INDEX(input,0,0) = 1;
    MAT_INDEX(input,1,0) = 1;

    forward(xor, input);
    float mse = cost(xor,td_x,td_y);
    printf("%f\n",mse);

    return 0;
}
