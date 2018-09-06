'''
input -> weights -> hidden layer 1 (activation function) ->
weights -> hidden layer 2(activation layer) -> weights -> output layer

feedforward
---------
comapre output to intended output -> cost function
cross entropy/ least mean square
---------
optimization function (optimizer)/ minimizes cost-
AdamOptimizer
Schotastic Gradient descent
AdamgGrad

Backpropagation
------

epoch = feedforward + backprop

'''
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_classes = 10
batch_size = 100
input_size = 784
n_nodes = [500, 500, 500, 400]

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_layer(input_size, output_size):
    layer = {'weights': tf.Variable(tf.random_normal([input_size, output_size])), 'biases': tf.Variable(tf.random_normal([output_size]))}
    return layer

def ann_model(n_nodes, input_size, output_size):

    hidden_layer = []

    layer = neural_layer(input_size, n_nodes[0])
    hidden_layer.append(layer)

    for x in range(len(n_nodes)-1):
        layer = neural_layer(n_nodes[x], n_nodes[x+1])
        hidden_layer.append(layer)

    layer = neural_layer(n_nodes[-1], output_size)
    hidden_layer.append(layer)

    return hidden_layer

#input * weights + biases
def ann_run(hidden_layer, data):

    val = []
    temp = tf.add(tf.matmul(data, hidden_layer[0]['weights']), hidden_layer[0]['biases'])
    temp = tf.nn.relu(temp)
    val.append(temp)

    n_iteration = len(hidden_layer)-2
    for x in range(n_iteration):
        temp = tf.add(tf.matmul(val[x], hidden_layer[x+1]['weights']), hidden_layer[x+1]['biases'])
        temp = tf.nn.relu(temp)
        val.append(temp)

    temp = tf.add(tf.matmul(val[-1], hidden_layer[-1]['weights']), hidden_layer[-1]['biases'])
    output = temp
    return output

def train_ann(hidden_layer, x):
    prediction = ann_run(hidden_layer, x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = prediction, labels = y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    n_epochs = 1

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(n_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                train_data, labels = mnist.train.next_batch(batch_size)

                writer = tf.summary.FileWriter("output1", sess.graph)
                opt, c = sess.run([optimizer, cost], feed_dict = {x: train_data, y: labels})
                writer.close()
                epoch_loss += c

            print("Epoch ", epoch, "completed out of ", n_epochs, " : epoch_loss = ", epoch_loss)

        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        #print(sess.run(correct))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        acc = accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
        print("Accuracy ", acc)

hidden_layer = ann_model(n_nodes, input_size, n_classes)
train_ann(hidden_layer, x)
