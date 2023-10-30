#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "nn.h"
#define DIM 2000

void single();
void multi();

int main(){
    multi();
    single();    
}

void single(){
    Matrix mat0 = mat_alloc(DIM,DIM);
    Matrix mat1 = mat_alloc(DIM,DIM);
    Matrix mat2 = mat_alloc(DIM,DIM);

    mat_fill(mat0,1);
    mat_fill(mat1,1);
    mat_fill(mat2,0);

    clock_t start, end;
    double cpu_time_used;

    start = clock();
    mat_dot(mat2, mat0,mat1);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Single threaded: %f\n", cpu_time_used);
}

void multi(){
    Matrix mat0 = mat_alloc(DIM,DIM);
    Matrix mat1 = mat_alloc(DIM,DIM);
    Matrix mat2 = mat_alloc(DIM,DIM);

    mat_fill(mat0,1);
    mat_fill(mat1,1);
    mat_fill(mat2,0);

    clock_t start, end;
    double cpu_time_used;

    start = clock();
    mat_dot_multithreaded(mat2, mat0, mat1);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Multi-threaded: %f\n", cpu_time_used);
}
