
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools
import numpy as np
tf.compat.v1.enable_eager_execution()

model = tf.keras.Sequential([
  tf.keras.layers.Dense(4, activation=tf.nn.relu, input_shape=(1,)),  # input shape required
  tf.keras.layers.Dense(4),
  tf.keras.layers.Dense(1)
])


def loss(model, x, y, training):
  # training=training is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  y_ = model(tf.slice(x,[0,0],[-1,1]), training=training)

  return tf.reduce_mean(tf.square(y - y_))


TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000
BS = 32
plot = False

inputs  = tf.random.normal(shape=[NUM_EXAMPLES,1])
noise   = tf.random.normal(shape=[NUM_EXAMPLES,1])
outputs = inputs * TRUE_W + TRUE_b + noise
in2 = np.hstack([inputs,inputs])
print(in2)


if plot:
    plt.scatter(inputs, outputs, c='b')
    plt.scatter(inputs, model(inputs), c='r')
    plt.show()

l = loss(model, in2, outputs, training=False)
print("Loss test: {}".format(l))

def grad(model, inputs, outputs,loss):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, outputs, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

def train_model(model,inputs,outputs,loss,learning_rate,N_epochs=20,batch_size=32,steps_per_epoch=None):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    train_loss_results = []
    print(steps_per_epoch)
    steps_per_epoch = steps_per_epoch or len(inputs) #if None, take num_samp steps
    print(steps_per_epoch)
    for epoch in range(N_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_inds = itertools.cycle(np.random.permutation(len(inputs)//batch_size+1)) #random shuffle inds
        n_steps = 0
        while n_steps<steps_per_epoch:
            i = next(epoch_inds)
            start = i*batch_size
            stop = i*batch_size + min(steps_per_epoch-n_steps,batch_size)
            loss_value, grads = grad(model, inputs[start:stop],
                                     outputs[start:stop],loss)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss
            n_steps += len(inputs[start:stop])
            if i == 31 or n_steps==steps_per_epoch:
                print(n_steps,i)
            
        train_loss_results.append(epoch_loss_avg.result())

        if epoch % 50 == 0:
            print("Epoch {:03d}: Loss: {:.3f}".format(epoch,epoch_loss_avg.result()))
    print("Epoch {:03d}: Loss: {:.3f}".format(epoch,epoch_loss_avg.result()))
    

# Collect the history of W-values and b-values to plot later
train_model(model, inputs, outputs, loss, learning_rate=0.1,steps_per_epoch=1000)

if plot:
    plt.scatter(inputs, outputs, c='b')
    plt.scatter(inputs, model(inputs), c='r')
    plt.show()
# Let's plot it all
#plt.plot(epochs, Ws, 'r',         epochs, bs, 'b')
#plt.plot([TRUE_W] * len(epochs), 'r--',        [TRUE_b] * len(epochs), 'b--')
#plt.legend(['W', 'b', 'True W', 'True b'])
#plt.show()




