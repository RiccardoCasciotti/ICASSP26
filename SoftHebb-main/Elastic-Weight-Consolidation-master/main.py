import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from IPython import display

# Load and preprocess MNIST using keras
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train.reshape(-1, 784) / 255.0, x_test.reshape(-1, 784) / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Create a simple MNIST dataset structure to mimic old code
class MNISTData:
    def __init__(self, images, labels):
        self._images = images
        self._labels = labels
        self.images = images
        self.labels = labels
        self.num_examples = images.shape[0]
        self._index = 0

    def next_batch(self, batch_size):
        if self._index + batch_size >= self.num_examples:
            self._index = 0
        batch_images = self.images[self._index:self._index + batch_size]
        batch_labels = self.labels[self._index:self._index + batch_size]
        self._index += batch_size
        return batch_images, batch_labels

class MNISTWrapper:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.train = MNISTData(x_train, y_train)
        self.validation = MNISTData(x_test, y_test)
        self.test = MNISTData(x_test, y_test)

mnist = MNISTWrapper(x_train, y_train, x_test, y_test)

# Updated permute function
def permute_mnist(mnist):
    perm_inds = np.random.permutation(mnist.train.images.shape[1])
    mnist2 = deepcopy(mnist)
    for split in ['train', 'validation', 'test']:
        dataset = getattr(mnist2, split)
        dataset._images = dataset.images[:, perm_inds]
    return mnist2

# Visualization
def plot_test_acc(plot_handles):
    plt.legend(handles=plot_handles, loc="center right")
    plt.xlabel("Iterations")
    plt.ylabel("Test Accuracy")
    plt.ylim(0, 1)
    plt.savefig("/projappl/project_462000765/casciott/DCASE25/SoftHebb-main/Elastic-Weight-Consolidation-master/Graph.png")
    # display.display(plt.gcf())
    # display.clear_output(wait=True)

# Placeholder training function using TensorFlow 2.x eager execution
def train_model(model, train_set, test_sets, num_iters=1000, disp_freq=50, lams=[0]):
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    for l, lam in enumerate(lams):
        test_accs = [np.zeros(num_iters // disp_freq) for _ in test_sets]

        for it in range(num_iters):
            batch_x, batch_y = train_set.train.next_batch(100)

            with tf.GradientTape() as tape:
                logits = model(batch_x, training=True)
                loss = loss_fn(batch_y, logits)
                if lam != 0:
                    loss += model.ewc_loss(lam)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if it % disp_freq == 0:
                plt.subplot(1, len(lams), l+1)
                plots = []
                for task_id, test in enumerate(test_sets):
                    preds = model(test.test.images, training=False)
                    acc = tf.keras.metrics.categorical_accuracy(test.test.labels, preds).numpy().mean()
                    index = it // disp_freq
                    test_accs[task_id][index] = acc
                    plot_h, = plt.plot(range(1, it+2, disp_freq), test_accs[task_id][:index+1], label=f"Task {chr(65+task_id)}")
                    plots.append(plot_h)

                plot_test_acc(plots)
                plt.title("Vanilla SGD" if lam == 0 else "EWC")
                plt.gcf().set_size_inches(len(lams)*5, 3.5)

# Replace with your actual model class converted to TF 2.x
class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10)

    def call(self, x, training=False):
        x = self.dense1(x)
        return self.dense2(x)

    def ewc_loss(self, lam):
        # Placeholder for EWC penalty logic
        return 0.0

    def compute_fisher(self, x_val, num_samples=200, plot_diffs=False):
        pass  # implement EWC Fisher logic here

    def star(self):
        pass  # implement variable saving for EWC

model = Model()

# Train Task A
train_model(model, mnist, [mnist], 800, 20)

# Save Fisher and star variables
model.compute_fisher(mnist.validation.images, num_samples=200, plot_diffs=False)
model.star()

# Train Task B
mnist2 = permute_mnist(mnist)
train_model(model, mnist2, [mnist, mnist2], 800, 20, lams=[0, 20])
model.compute_fisher(mnist2.validation.images, num_samples=200, plot_diffs=False)
model.star()

# Task C
mnist3 = permute_mnist(mnist)
