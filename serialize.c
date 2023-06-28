// File format:
// <magic>
// <number of layers>
// For each layer:
//   <number of layer inputs>
//   <number of layer neurons>
//   <weights>
//   <bias>
//   <activation>

typedef struct {
  int num_inputs;
  int num_neurons;
  double* weights; // array of size num_inputs * num_neurons
  double* biases; // array of size num_neurons
  char activation[50]; // name of the activation function
} Layer;

typedef struct {
  char magic[8]; // magic string
  int num_layers;
  Layer* layers;
} NeuralNetwork;

void save_neural_network(const char* filename, NeuralNetwork* net) {
  FILE* file = fopen(filename, "wb");
  if (file == NULL) {
    printf("Could not open file for writing: %s\n", filename);
    return;
  }

  fwrite(net->magic, sizeof(net->magic), 1, file);
  fwrite(&net->num_layers, sizeof(net->num_layers), 1, file);
  
  for (int i = 0; i < net->num_layers; ++i) {
    Layer* layer = &net->layers[i];
    fwrite(&layer->num_inputs, sizeof(layer->num_inputs), 1, file);
    fwrite(&layer->num_neurons, sizeof(layer->num_neurons), 1, file);
    fwrite(layer->weights, sizeof(double), layer->num_inputs * layer->num_neurons, file);
    fwrite(layer->biases, sizeof(double), layer->num_neurons, file);
    fwrite(layer->activation, sizeof(layer->activation), 1, file);
  }
  
  fclose(file);
}

NeuralNetwork* load_neural_network(const char* filename) {
  FILE* file = fopen(filename, "rb");
  if (file == NULL) {
    printf("Could not open file for reading: %s\n", filename);
    return NULL;
  }

  NeuralNetwork* net = malloc(sizeof(NeuralNetwork));
  
  fread(net->magic, sizeof(net->magic), 1, file);
  if (strcmp(net->magic, "NNETV1.0") != 0) {
    printf("Invalid magic string: %s\n", net->magic);
    free(net);
    return NULL;
  }
  
  fread(&net->num_layers, sizeof(net->num_layers), 1, file);
  net->layers = malloc(sizeof(Layer) * net->num_layers);
  
  for (int i = 0; i < net->num_layers; ++i) {
    Layer* layer = &net->layers[i];
    fread(&layer->num_inputs, sizeof(layer->num_inputs), 1, file);
    fread(&layer->num_neurons, sizeof(layer->num_neurons), 1, file);
    layer->weights = malloc(sizeof(double) * layer->num_inputs * layer->num_neurons);
    layer->biases = malloc(sizeof(double) * layer->num_neurons);
    fread(layer->weights, sizeof(double), layer->num_inputs * layer->num_neurons, file);
    fread(layer->biases, sizeof(double), layer->num_neurons, file);
    fread(layer->activation, sizeof(layer->activation), 1, file);
  }
  
  fclose(file);
  
  return net
}