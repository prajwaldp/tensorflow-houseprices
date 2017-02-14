'''
A simple python script using tensorflow to create a artificial neural
network model to predict house prices
'''

import pandas as pd
import tensorflow as tf


# preparing the dataframe
df = pd.read_csv('data.csv')

# use only the first 10 rows and `area`
# and `bathrooms` columns
df = df.loc[0:9, ('area', 'bathrooms')]

# set up labels and one-hot encode them
df.loc[:, ('y1')] = [1, 1, 1, 0, 0, 1, 0, 1, 1, 1]
df.loc[:, ('y2')] = df.y1 == 0
df.loc[:, ('y2')] = df.y2.astype(int)

# convert dataframes to numpy arrays
input_x = df.loc[:, ('area', 'bathrooms')].as_matrix()
input_y = df.loc[:, ('y1', 'y2')].as_matrix()

# set up hyperparameters
learning_rate = 0.00001
training_epochs = 200
display_step = 50
n_samples = input_y.size

# set up the graph
X = tf.placeholder(tf.float32, [None, 2], name='X')
y = tf.placeholder(tf.float32, [None, 2], name='y')

# weights and biases
W = tf.Variable(tf.zeros([2, 2]))
b = tf.Variable(tf.zeros([2]))

# output of the neural network
y_values = tf.add(tf.matmul(X, W), b)
y_ = tf.nn.softmax(y_values)

# cost function (mean squared error)
cost = tf.reduce_sum(tf.pow(y - y_, 2)) / (2 * n_samples)

# gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# accuracy
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# set up summaries for tensorboard
tf.summary.scalar('cost', cost)
tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logs")

# start graph
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(training_epochs):

        sess.run(optimizer, feed_dict={X: input_x, y: input_y})

        if ((i + 1) % display_step == 0):

            summary, cc = sess.run([merged, cost], feed_dict={
                                   X: input_x, y: input_y})
            writer.add_summary(summary, i)
            print("Training Step {}: Cost: {}".format(i + 1, cc))

    print("\nOptimization Completed")
    training_cost = sess.run(cost, feed_dict={X: input_x, y: input_y})

    print("Training cost {}\nW = {}\nb={}".format(
        training_cost, sess.run(W), sess.run(b)))
