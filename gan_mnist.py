
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


from tensorflow.examples.tutorials.mnist import input_data


# In[3]:


import numpy as np
import matplotlib.pyplot as plt


# In[5]:


mnist = input_data.read_data_sets("/tmp/data/", one_hot= True)


# In[84]:


plt.imshow(mnist.train.images[4].reshape(28, 28), cmap = "Greys")


# In[138]:


# This will reset the computation graph, if some variables are disallowed, run this once
tf.reset_default_graph()


# In[139]:


def generator(z, reuse= None):
    with tf.variable_scope("gen", reuse= reuse):
        # Leak factor
        alpha = 0.01

        hidden1 = tf.layers.dense(inputs = z, units = 128)
        hidden1 = tf.maximum(hidden1, alpha*hidden1)

        hidden2 = tf.layers.dense(inputs = hidden1, units = 128)
        hidden2 = tf.maximum(hidden2, alpha*hidden2)

        output = tf.layers.dense(inputs = hidden2, units = 25, activation = tf.nn.tanh)

        return output


# In[140]:


def discriminator(X, reuse=None):
    #print(real_images.get_shape().as_list())
    with tf.variable_scope("dis", reuse= reuse):
        alpha = 0.01

        hidden1 = tf.layers.dense(inputs= X, units= 128)
        hidden1 = tf.maximum(hidden1, hidden1*alpha)

        hidden2 = tf.layers.dense(inputs= hidden1, units= 128)
        hidden2 = tf.maximum(hidden2, hidden2*alpha)

        logits = tf.layers.dense(inputs= hidden2, units= 1)
        output = tf.sigmoid(logits)

        return output, logits


# In[141]:


real_images = tf.placeholder(tf.float32, shape= [None, 784])
z = tf.placeholder(tf.float32, shape= [None, 100])


# In[142]:


G = generator(z)


# In[143]:


D_output_real, D_logits_real = discriminator(real_images)


# In[ ]:


D_output_fake, D_logits_fake = discriminator(G, reuse= True)


# In[145]:


def loss_func(logits_in, labels_in):
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits= logits_in, labels= labels_in))
    return cost


# In[146]:


D_loss_real = loss_func(D_logits_real, tf.ones_like(D_logits_real))
D_loss_fake = loss_func(D_logits_fake, tf.zeros_like(D_logits_real))


# In[147]:


D_loss = D_loss_real + D_loss_fake


# In[148]:


G_loss = loss_func(D_logits_fake, tf.ones_like(D_logits_fake))


# In[191]:


tvars = tf.trainable_variables()

d_vars = []
g_vars = []

for i in range(len(tvars)):
    if tvars[i].name[:3] == "gen":
        g_vars.append(tvars[i])
    else:
        d_vars.append(tvars[i])


# In[186]:


learning_rate = 0.001


# In[194]:


D_trainer = tf.train.AdamOptimizer(learning_rate).minimize(D_loss, var_list = d_vars)


# In[195]:


G_trainer = tf.train.AdamOptimizer(learning_rate).minimize(G_loss, var_list = g_vars)


# In[236]:


epochs = 10
batch_size = 100


# In[237]:


init = tf.global_variables_initializer()


# In[238]:


samples = []

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(epochs):
        num_batches = int(mnist.train.num_examples/batch_size)

        for i in range(num_batches):
            batch = mnist.train.next_batch(batch_size)
            batch_images = batch[0].reshape((batch_size, 784))
            batch_images = batch_images*2-1

            batch_z = np.random.uniform(-1, 1, size= (batch_size, 100))

            _ = sess.run(D_trainer, feed_dict={real_images:batch_images, z: batch_z})
            _ = sess.run(G_trainer, feed_dict={z: batch_z})

        print("epoch ", epoch , " completed.")
        sample_z = np.random.uniform(-1, 1, size= (1, 100))
        gen_sample = sess.run(generator(z, reuse=True), feed_dict= {z:sample_z})

        samples.append(gen_sample)


# In[260]:


plt.imshow(samples[9].reshape(28,28), cmap="Greys")
plt.show()
