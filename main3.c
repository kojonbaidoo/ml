#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define LEARNING_RATE 10
#define EPS 1e-3
#define EPOCHS 2000000
#define TRAINING_SAMPLE_SIZE 4
#define TESTING_SAMPLE_SIZE 50

float training_data[TRAINING_SAMPLE_SIZE][3] = {
        {0,0,0},
        {0,1,1},
        {1,0,1},
        {1,1,0}
    };
float w0,w1,w2,w3,w4,w5;
float bias_0,bias_1,bias_2;

void init_training_data(){

}

void init_weight_and_bias(){
    w0 = 0.1;
    w1 = 1.6;
    w2 = 0.1;
    w3 = 1.6;
    w4 = 0.1;
    w5 = 1.6;
    bias_0 = 0;
    bias_1 = 0;
    bias_2 = 0;
}

float sigmoidf(float value){
    return 1.0 / (1.0 + expf(-value));
}

float backprop_weight_0(float w0, float w1, float w2,float w3, float w4,float w5, float bias_0,float bias_1,float bias_2){
    float weight_deriv = 0;
    float sigmoid_1;
    float sigmoid_A;
    float sigmoid_B;
    for(int iter = 0;iter < TRAINING_SAMPLE_SIZE;iter++){
        sigmoid_A = sigmoidf(w0*training_data[iter][0] + w1*training_data[iter][1] + bias_0);
        sigmoid_B = sigmoidf(w2*training_data[iter][0] + w3*training_data[iter][1] + bias_1);
        sigmoid_1 = sigmoidf(w4*sigmoid_A + w5*sigmoid_B + bias_2);

        weight_deriv += -2 * (training_data[iter][2] - sigmoid_1) * (sigmoid_1 * (1 - sigmoid_1)) * w4 * (sigmoid_A * (1 - sigmoid_A)) * training_data[iter][0];
        // weight_deriv += (-2 * training_data[iter][index]) * (training_data[iter][2] - (w0*training_data[iter][0] + w1*training_data[iter][1] + bias));
    }
    return weight_deriv / TRAINING_SAMPLE_SIZE;
}

float backprop_weight_1(float w0, float w1, float w2,float w3, float w4,float w5, float bias_0,float bias_1,float bias_2){
    float weight_deriv = 0;
    float sigmoid_1;
    float sigmoid_A;
    float sigmoid_B;
    for(int iter = 0;iter < TRAINING_SAMPLE_SIZE;iter++){
        sigmoid_A = sigmoidf(w0*training_data[iter][0] + w1*training_data[iter][1] + bias_0);
        sigmoid_B = sigmoidf(w2*training_data[iter][0] + w3*training_data[iter][1] + bias_1);
        sigmoid_1 = sigmoidf(w4*sigmoid_A + w5*sigmoid_B + bias_2);

        weight_deriv += -2 * (training_data[iter][2] - sigmoid_1) * (sigmoid_1 * (1 - sigmoid_1)) * w4 * (sigmoid_A * (1 - sigmoid_A)) * training_data[iter][1];
        // weight_deriv += (-2 * training_data[iter][index]) * (training_data[iter][2] - (w0*training_data[iter][0] + w1*training_data[iter][1] + bias));
    }
    return weight_deriv / TRAINING_SAMPLE_SIZE;
}

float backprop_bias_0(float w0, float w1, float w2,float w3, float w4,float w5, float bias_0,float bias_1,float bias_2){
    float bias_deriv = 0;
    float sigmoid_1;
    float sigmoid_A;
    float sigmoid_B;
    for(int iter = 0;iter < TRAINING_SAMPLE_SIZE;iter++){
        sigmoid_A = sigmoidf(w0*training_data[iter][0] + w1*training_data[iter][1] + bias_0);
        sigmoid_B = sigmoidf(w2*training_data[iter][0] + w3*training_data[iter][1] + bias_1);
        sigmoid_1 = sigmoidf(w4*sigmoid_A + w5*sigmoid_B + bias_2);

        bias_deriv += -2 * (training_data[iter][2] - sigmoid_1) * (sigmoid_1 * (1 - sigmoid_1)) * w4 * (sigmoid_A * (1 - sigmoid_A));
        // weight_deriv += (-2 * training_data[iter][index]) * (training_data[iter][2] - (w0*training_data[iter][0] + w1*training_data[iter][1] + bias));
    }
    return bias_deriv / TRAINING_SAMPLE_SIZE;
}

float backprop_weight_2(float w0, float w1, float w2,float w3, float w4,float w5, float bias_0,float bias_1,float bias_2){
    float weight_deriv = 0;
    float sigmoid_1;
    float sigmoid_A;
    float sigmoid_B;
    for(int iter = 0;iter < TRAINING_SAMPLE_SIZE;iter++){
        sigmoid_A = sigmoidf(w0*training_data[iter][0] + w1*training_data[iter][1] + bias_0);
        sigmoid_B = sigmoidf(w2*training_data[iter][0] + w3*training_data[iter][1] + bias_1);
        sigmoid_1 = sigmoidf(w4*sigmoid_A + w5*sigmoid_B + bias_2);

        weight_deriv += -2 * (training_data[iter][2] - sigmoid_1) * (sigmoid_1 * (1 - sigmoid_1)) * w5 * (sigmoid_B * (1 - sigmoid_B)) * training_data[iter][0];
        // weight_deriv += (-2 * training_data[iter][index]) * (training_data[iter][2] - (w0*training_data[iter][0] + w1*training_data[iter][1] + bias));
    }
    return weight_deriv / TRAINING_SAMPLE_SIZE;
}

float backprop_weight_3(float w0, float w1, float w2,float w3, float w4,float w5, float bias_0,float bias_1,float bias_2){
    float weight_deriv = 0;
    float sigmoid_1;
    float sigmoid_A;
    float sigmoid_B;
    for(int iter = 0;iter < TRAINING_SAMPLE_SIZE;iter++){
        sigmoid_A = sigmoidf(w0*training_data[iter][0] + w1*training_data[iter][1] + bias_0);
        sigmoid_B = sigmoidf(w2*training_data[iter][0] + w3*training_data[iter][1] + bias_1);
        sigmoid_1 = sigmoidf(w4*sigmoid_A + w5*sigmoid_B + bias_2);

        weight_deriv += -2 * (training_data[iter][2] - sigmoid_1) * (sigmoid_1 * (1 - sigmoid_1)) * w5 * (sigmoid_B * (1 - sigmoid_B)) * training_data[iter][1];
        // weight_deriv += (-2 * training_data[iter][index]) * (training_data[iter][2] - (w0*training_data[iter][0] + w1*training_data[iter][1] + bias));
    }
    return weight_deriv / TRAINING_SAMPLE_SIZE;
}

float backprop_bias_1(float w0, float w1, float w2,float w3, float w4,float w5, float bias_0,float bias_1,float bias_2){
    float bias_deriv = 0;
    float sigmoid_1;
    float sigmoid_A;
    float sigmoid_B;
    for(int iter = 0;iter < TRAINING_SAMPLE_SIZE;iter++){
        sigmoid_A = sigmoidf(w0*training_data[iter][0] + w1*training_data[iter][1] + bias_0);
        sigmoid_B = sigmoidf(w2*training_data[iter][0] + w3*training_data[iter][1] + bias_1);
        sigmoid_1 = sigmoidf(w4*sigmoid_A + w5*sigmoid_B + bias_2);

        bias_deriv += -2 * (training_data[iter][2] - sigmoid_1) * (sigmoid_1 * (1 - sigmoid_1)) * w5 * (sigmoid_B * (1 - sigmoid_B));
        // weight_deriv += (-2 * training_data[iter][index]) * (training_data[iter][2] - (w0*training_data[iter][0] + w1*training_data[iter][1] + bias));
    }
    return bias_deriv / TRAINING_SAMPLE_SIZE;
}

float backprop_weight_4(float w0, float w1, float w2,float w3, float w4,float w5, float bias_0,float bias_1,float bias_2){
    float weight_deriv = 0;
    float sigmoid_1;
    float sigmoid_A;
    float sigmoid_B;
    for(int iter = 0;iter < TRAINING_SAMPLE_SIZE;iter++){
        sigmoid_A = sigmoidf(w0*training_data[iter][0] + w1*training_data[iter][1] + bias_0);
        sigmoid_B = sigmoidf(w2*training_data[iter][0] + w3*training_data[iter][1] + bias_1);
        sigmoid_1 = sigmoidf(w4*sigmoid_A + w5*sigmoid_B + bias_2);

        weight_deriv += -2 * (training_data[iter][2] - sigmoid_1) * (sigmoid_1 * (1 - sigmoid_1)) * sigmoid_A;
        // weight_deriv += (-2 * training_data[iter][index]) * (training_data[iter][2] - (w0*training_data[iter][0] + w1*training_data[iter][1] + bias));
    }
    return weight_deriv / TRAINING_SAMPLE_SIZE;
}

float backprop_weight_5(float w0, float w1, float w2,float w3, float w4,float w5, float bias_0,float bias_1,float bias_2){
    float weight_deriv = 0;
    float sigmoid_1;
    float sigmoid_A;
    float sigmoid_B;
    for(int iter = 0;iter < TRAINING_SAMPLE_SIZE;iter++){
        sigmoid_A = sigmoidf(w0*training_data[iter][0] + w1*training_data[iter][1] + bias_0);
        sigmoid_B = sigmoidf(w2*training_data[iter][0] + w3*training_data[iter][1] + bias_1);
        sigmoid_1 = sigmoidf(w4*sigmoid_A + w5*sigmoid_B + bias_2);

        weight_deriv += -2 * (training_data[iter][2] - sigmoid_1) * (sigmoid_1 * (1 - sigmoid_1)) * sigmoid_B;
        // weight_deriv += (-2 * training_data[iter][index]) * (training_data[iter][2] - (w0*training_data[iter][0] + w1*training_data[iter][1] + bias));
    }
    return weight_deriv / TRAINING_SAMPLE_SIZE;
}

float backprop_bias_2(float w0, float w1, float w2,float w3, float w4,float w5, float bias_0,float bias_1,float bias_2){
    float bias_deriv = 0;
    float sigmoid_1;
    float sigmoid_A;
    float sigmoid_B;
    for(int iter = 0;iter < TRAINING_SAMPLE_SIZE;iter++){
        sigmoid_A = sigmoidf(w0*training_data[iter][0] + w1*training_data[iter][1] + bias_0);
        sigmoid_B = sigmoidf(w2*training_data[iter][0] + w3*training_data[iter][1] + bias_1);
        sigmoid_1 = sigmoidf(w4*sigmoid_A + w5*sigmoid_B + bias_2);

        bias_deriv += -2 * (training_data[iter][2] - sigmoid_1) * (sigmoid_1 * (1 - sigmoid_1));
        // weight_deriv += (-2 * training_data[iter][index]) * (training_data[iter][2] - (w0*training_data[iter][0] + w1*training_data[iter][1] + bias));
    }
    return bias_deriv / TRAINING_SAMPLE_SIZE;
}

float cost(float w0, float w1, float w2,float w3, float w4,float w5, float bias_0,float bias_1,float bias_2){
    float actual;
    float expected;
    float error;
    float distance;
    float mse = 0.0;

    float sigmoid_1;
    float sigmoid_A;
    float sigmoid_B;

    for(int iter = 0;iter < TRAINING_SAMPLE_SIZE;iter++){
        sigmoid_A = sigmoidf(w0*training_data[iter][0] + w1*training_data[iter][1] + bias_0);
        sigmoid_B = sigmoidf(w2*training_data[iter][0] + w3*training_data[iter][1] + bias_1);
        sigmoid_1 = sigmoidf(w4*sigmoid_A + w5*sigmoid_B + bias_2);

        expected = training_data[iter][2];
        actual = sigmoid_1;
        distance = expected - actual;
        error = distance * distance;
        mse += error;
    }
    return (mse / TRAINING_SAMPLE_SIZE);
}

void train(){
    float error;
    float dW0,dW1,dW2,dW3,dW4,dW5;
    float dB0,dB1,dB2;
    
    for(int epoch = 0; epoch < EPOCHS; epoch++){
        dW0 = backprop_weight_0(w0,w1,w2,w3,w4,w5,bias_0,bias_1,bias_2);
        dW1 = backprop_weight_1(w0,w1,w2,w3,w4,w5,bias_0,bias_1,bias_2);
        dW2 = backprop_weight_2(w0,w1,w2,w3,w4,w5,bias_0,bias_1,bias_2);
        dW3 = backprop_weight_3(w0,w1,w2,w3,w4,w5,bias_0,bias_1,bias_2);
        dW4 = backprop_weight_4(w0,w1,w2,w3,w4,w5,bias_0,bias_1,bias_2);
        dW5 = backprop_weight_5(w0,w1,w2,w3,w4,w5,bias_0,bias_1,bias_2);

        dB0 = backprop_bias_0(w0,w1,w2,w3,w4,w5,bias_0,bias_1,bias_2);
        dB1 = backprop_bias_1(w0,w1,w2,w3,w4,w5,bias_0,bias_1,bias_2);
        dB2 = backprop_bias_2(w0,w1,w2,w3,w4,w5,bias_0,bias_1,bias_2);

        w0 -= LEARNING_RATE*dW0;
        w1 -= LEARNING_RATE*dW1;
        w2 -= LEARNING_RATE*dW2;
        w3 -= LEARNING_RATE*dW3;
        w4 -= LEARNING_RATE*dW4;
        w5 -= LEARNING_RATE*dW5;

        bias_0 -= LEARNING_RATE*dB0;
        bias_1 -= LEARNING_RATE*dB1;
        bias_2 -= LEARNING_RATE*dB2;
        
        error = cost(w0,w1,w2,w3,w4,w5,bias_0,bias_1,bias_2);

        printf("W0: %.2f\t",w0);
        printf("W1: %.2f\t",w1);
        printf("B0: %.2f\t",bias_0);

        printf("W2: %.2f\t",w2);
        printf("W3: %.2f\t",w3);
        printf("B1: %.2f\t",bias_1);

        printf("W3: %.2f\t",w4);
        printf("W4: %.2f\t",w5);
        printf("B2: %.2f\t",bias_2);
        printf("Error: %.2f\n",error);

        if(error < 0.0001){break;}
    }
}

float predict(float w0, float w1, float w2,float w3, float w4,float w5, float bias_0,float bias_1,float bias_2, float x0, float x1){
    float sigmoid_1;
    float sigmoid_A;
    float sigmoid_B;

    sigmoid_A = sigmoidf(w0*x0 + w1*x1 + bias_0);
    sigmoid_B = sigmoidf(w2*x0 + w3*x1 + bias_1);
    sigmoid_1 = sigmoidf(w4*sigmoid_A + w5*sigmoid_B + bias_2);

    return sigmoid_1;
}

int main(){
    init_training_data();
    init_weight_and_bias();
    train();
    for(int i = 0; i < TRAINING_SAMPLE_SIZE;i++){
        printf("\n%.1f | %.1f == %.4f\n",training_data[i][0],training_data[i][1],predict(w0,w1,w2,w3,w4,w5,bias_0,bias_1,bias_2,training_data[i][0],training_data[i][1]));
    }
    // init_testing_data();
    // test();
}