# lstm_tensorflow
Use tensorflow to build a lstm model.

It refers to the paper of 'LSTM-BASED DEEP LEARNING MODELS FOR NON-FACTOID ANSWER SELECTION' written by Ming Tan, etc.

Based on Tensorflow in python, a model with lstm and attention mechanism is implemented.

I build a biLSTM model and train it for QA system. The parameters for train model are           
    num_nodes = 141        # number of hidden layer units         
    embedding_size = 100            
    batch_size = 100s           
    seq_len = 200           
    loss_margin = 0.1        # 0.2, 0.3         
    learning_rate = 0.1      # 0.01         
    num_epoch = 10          
    eval_every = 20     
    ratio = batch_size       # It is for test.      
    test_size = 20      
Just ignore the test texts in the training model bilstm.py. 
The training process is as follows: 
  - First I set margin=0.1.
  - Then tune margin=0.2, when the loss of training decreases a lot, training accuracy increases a lot while the test accuracy is still low. It is helpful for params tuning.
  - Later I tune margin=0.3 as the loss decreases much, however, it helps little. 

The test params are
    num_nodes = 141           # number of hidden layer units
    embedding_size = 100      # 100
    batch_size = 500          
    seq_len = 200
    loss_margin = 0.1
    ratio = batch_size        # for test  == batch_size
    test_size = 100 
These params mean there are test_size=100 test cases in the test sets. For every test case there are ratio=batch_size=500 answers in answer pool, and only one correct answer exists. 

The best test result is: accuracy = 0.85, when test size = 100, test ratio = 500.

