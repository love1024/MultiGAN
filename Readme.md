# Training
1. Put training images in training_data folder
2. cmd python multigan_train.py
3. After training, pretrained folder will contain distributions
4. GPU is required for training

# Testing
1. Put your test image in test_image folder
2. cmd python multigan_test.py
3. The generated result will be inside test_results folder
4. Parallel siamese neural network only works on linux, otherwise sequential