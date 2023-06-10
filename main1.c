#include <stdio.h>
#include <stdlib.h>

#define LEARNING_RATE 0.0003
#define EPS 1e-3
#define EPOCHS 20000
#define TRAINING_SAMPLE_SIZE 100
#define TESTING_SAMPLE_SIZE 50

float training_data[TRAINING_SAMPLE_SIZE][2];
float testing_data[TESTING_SAMPLE_SIZE][2];
float weight;
float bias;

void init_training_data(){
    for(int item = 0;item < TRAINING_SAMPLE_SIZE;item++){
        training_data[item][0] = item;
        training_data[item][1] = item*2;
    }
}

void init_testing_data(){
    for(int item = 0;item < TESTING_SAMPLE_SIZE;item++){
        testing_data[item][0] = item*20;
        testing_data[item][1] = item*2*20;
    }
}

void init_weight_and_bias(){
    weight = 1.2;
    bias = 0.7;
}

float backprop_weight(float weight, float bias){
    float weight_deriv = 0;
    for(int iter = 0;iter < TRAINING_SAMPLE_SIZE;iter++){
        weight_deriv += (-2 * training_data[iter][0]) * (training_data[iter][1] - (weight*training_data[iter][0] + bias));
    }
    return weight_deriv / TRAINING_SAMPLE_SIZE;

}

float backprop_bias(float weight, float bias){
    float bias_deriv = 0;
    for(int iter = 0;iter < TRAINING_SAMPLE_SIZE;iter++){
        bias_deriv += -2 * (training_data[iter][1] - (weight*training_data[iter][0] + bias));
    }
    return bias_deriv / TRAINING_SAMPLE_SIZE;
}

float cost(float weight, float bias){
    float actual;
    float expected;
    float error;
    float distance;
    float mse = 0.0;

    for(int iter = 0;iter < TRAINING_SAMPLE_SIZE;iter++){
        expected = training_data[iter][1];
        actual = training_data[iter][0] * weight + bias;
        distance = expected - actual;
        error = distance * distance;
        mse += error;
    }
    return (mse / TRAINING_SAMPLE_SIZE);
}

void train(){
    float error;
    float dW;
    float dB;
    for(int epoch = 0; epoch < EPOCHS; epoch++){
        dW = backprop_weight(weight,bias);
        dB = backprop_bias(weight,bias);

        weight -= LEARNING_RATE*dW;
        bias -= LEARNING_RATE*dB;
        error = cost(weight,bias);

        printf("dW: %.2f\t", dW);
        printf("dB: %.2f\t", dB);
        printf("Weight: %.2f\t",weight);
        printf("Bias: %.2f\t",bias);
        printf("Error: %.2f\n",error);
    }
}

void test(){
    float actual;
    float expected;
    float error;
    float distance;

    for(int iter = 0;iter < TESTING_SAMPLE_SIZE;iter++){
        expected = testing_data[iter][1];
        actual = testing_data[iter][0] * weight + bias;
        distance = expected - actual;
        error = distance * distance * distance;
        printf("Expected: %f\t",expected);
        printf("Actual: %f\t",actual);
        printf("Error: %f\t\t",error);
        printf("weight: %f\n",weight);
    }
}
int main(){
    init_training_data();
    init_weight_and_bias();
    train();
}