#define NN_IMPLEMENTATION
#include "nn.h"
#include <assert.h>

#define MAT_INDEX(MAT,ROW,COL) MAT.vals[(ROW) * (MAT.cols) + (COL)]
#define EPOCHS 1
#define TRAINING_SAMPLE_SIZE 4

float training_data[TRAINING_SAMPLE_SIZE][3] = {
    {0,0,0},
    {0,1,1},
    {1,0,1},
    {1,1,0},
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

    Layer layer0 = layer_alloc(2,16,SIGMOID);
    Layer layer1 = layer_alloc(16,16,SIGMOID);
    Layer layer2 = layer_alloc(16,1,SIGMOID);

    MLP mlp = mlp_alloc(3);
    
    mlp_add(&mlp,layer0);
    mlp_add(&mlp,layer1);
    mlp_add(&mlp,layer2);

    mat_print(mlp_cost(mlp,td_x,td_y));
    mlp_train(mlp,td_x,td_y,1,10000);
    mat_print(mlp_cost(mlp,td_x,td_y));

    save_neural_network("net.mat", &mlp);
    
    MLP *net = load_neural_network("net.mat");
    mat_print(mlp_cost(*net, td_x, td_y));

    return 0;
}
