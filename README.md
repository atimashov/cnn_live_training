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
* parameters
* statistics
* activations

**parameters**  
This table contains *only one row* with current parameters for training CNN model.  
When we change any parameters in our dashboard (file `board.py`), this data will be updated in *parameters* SQL table. The table contains following columns:
* optimizer
* learning_rate
* weight_decay
* dropout
* dt_updates
* stop_train

The first *four* columns are parameters of training. The columns *dt_updates* indicates timeseries, when data was modified. 
Finally, *stop_train* is a boolean variable that indicates should we stop training or continue it.

**statistics**
* dt_started
* model_name
* epoch
* step
* optimizer
* learning_rate
* weight_decay
* dropout
* dt
* train_loss
* train_accuracy
* validate_loss
* validate_accuracy

**activations**  
This table contains current *distribution of weights* in activation maps for all *convolutional* and *fully connected* layers. The table contains following columns:
* nn_part
* layer_type
* number
* weights
* num_weights

The column *nn_part* can be either features or classifier; *layer_type* can be conv or fc, which indicates convolutional and fully connected layers respectively; *number* us the layer number in the model part. (features/classifier)  
The column *weights* indicates the average value of weights in particular bin of; *num_weights* indicates number of weight from the particular bin. 


### Steps to run the system
1. Install required packages from `requirements.txt`;
2. Run training process by `python train.py`;
3. Run monitoring dashboard by `python board.py`;
