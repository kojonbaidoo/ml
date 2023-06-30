#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define LEARNING_RATE 0.1
#define EPS 1e-3
#define EPOCHS 1000
#define TRAINING_SAMPLE_SIZE 4
#define TESTING_SAMPLE_SIZE 50

float training_data[TRAINING_SAMPLE_SIZE][3] = {
        {0,0,1},
        {0,1,1},
        {1,0,1},
        {1,1,0}
    };
float w0;
float w1;
float bias;

void init_training_data(){

}

void init_weight_and_bias(){
    w0 = 0.1;
    w1 = 1.6;
    bias = 0;
}

float sigmoidf(float value){
    return 1.0 / (1.0 + expf(-value));
}

float backprop_weight(float w0, float w1, float bias, int index){
    float weight_deriv = 0;
    float sigmoid;
    for(int iter = 0;iter < TRAINING_SAMPLE_SIZE;iter++){
        sigmoid = sigmoidf(w0*training_data[iter][0] + w1*training_data[iter][1] + bias);
        weight_deriv += (-2 * training_data[iter][index]) * (training_data[iter][2] - sigmoid) * (sigmoid * (1 - sigmoid));
        // weight_deriv += (-2 * training_data[iter][index]) * (training_data[iter][2] - (w0*training_data[iter][0] + w1*training_data[iter][1] + bias));
    }
    return weight_deriv / TRAINING_SAMPLE_SIZE;

}

float backprop_bias(float w0, float w1, float bias){
    float bias_deriv = 0;
    float sigmoid;
    for(int iter = 0;iter < TRAINING_SAMPLE_SIZE;iter++){
        sigmoid = sigmoidf(w0*training_data[iter][0] + w1*training_data[iter][1] + bias);
        bias_deriv += (-2) * (training_data[iter][2] - sigmoid) * (sigmoid * (1 - sigmoid));
        // bias_deriv += -2 * (training_data[iter][2] - (w0*training_data[iter][0] + w1*training_data[iter][1] + bias));
    }
    return bias_deriv / TRAINING_SAMPLE_SIZE;
}

float cost(float w0,float w1, float bias){
    float actual;
    float expected;
    float error;
    float distance;
    float mse = 0.0;

    for(int iter = 0;iter < TRAINING_SAMPLE_SIZE;iter++){
        expected = training_data[iter][2];
        actual = sigmoidf(w0*training_data[iter][0] + w1*training_data[iter][1] + bias);
        distance = expected - actual;
        error = distance * distance;
        mse += error;
    }
    return (mse / TRAINING_SAMPLE_SIZE);
}

void train(){
    float error;
    float dW0;
    float dW1;
    float dB;
    for(int epoch = 0; epoch < EPOCHS; epoch++){
        dW0 = backprop_weight(w0,w1,bias,0);
        dW1 = backprop_weight(w0,w1,bias,1);
        dB = backprop_bias(w0,w1,bias);

        w0 -= LEARNING_RATE*dW0;
        w1 -= LEARNING_RATE*dW1;

        bias -= LEARNING_RATE*dB;
        
        error = cost(w0,w1,bias);

        printf("dW0: %.2f\t", dW0);
        printf("dW1: %.2f\t", dW1);
        printf("dB: %.2f\t", dB);
        printf("W0: %.2f\t",w0);
        printf("W1: %.2f\t",w1);
        printf("Bias: %.2f\t",bias);
        printf("Error: %.2f\n",error);

        if(error < 0.001){break;}
    }
}

float predict(float w0, float w1, float bias, float x0, float x1){
    return sigmoidf(w0*x0 + w1*x1 + bias);
}

int main(){
    init_training_data();
    init_weight_and_bias();
    train();
    for(int i = 0; i < TRAINING_SAMPLE_SIZE;i++){
        printf("\n%.1f | %.1f == %.4f\n",training_data[i][0],training_data[i][1],predict(w0,w1,bias,training_data[i][0],training_data[i][1]));
    }
    // init_testing_data();
    // test();
}