# Live CNN Training Dashboard
![alt text](https://github.com/atimashov/cnn_live_training/blob/main/dash.png?raw=true)
This repository represents dashboard for live training CNN model. 
The idea is to be able to train CNN model, change parameters "on the way" and see immediate result. It might be helpfult to nurture "feeling" of training neural network.

### Available features (real-time)
* Loss function;
* Accuracy;
* Activation maps distributions;

### Available tuning parameters
* change optimizer (Adam & SGD + Nesterov);
* change learning rate (step is 0.00005);
* change weight decay;
* change dropout (20% - 80%, step is 1%)

### Preparation
Before running we have to create DB _dl_playground_ in _PostgreSQL_ with the schema "cnn_live_training" that contains three following tables:
* activations
* parameters
* statistics

### Steps to run the system
1. Install required packages from `requirements.txt`;
2. Run training process by `python train.py`;
3. Run monitoring dashboard by `python board.py`;
