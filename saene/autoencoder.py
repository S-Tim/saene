""" Stacked Autoencoder using tensorflow

Stacked Autoencoder Reference:
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/autoencoder.py

Author: Tim Silhan
"""

from uuid import uuid4
import tensorflow as tf
import numpy as np

from utils.comparison_plotter import plot_comparison
from utils.time_me import time_me
from utils.history_saver import HistorySaver

class Autoencoder:
    """ Stacked Autoencoder

    Autoencoders can be created from a list of layer sizes. Layers can then
    be appended to the autoencoder. Further it is possible to manage which
    layers are trainable and which are not, using the *freeze* methods and
    the *get_trainable_layers* function.

    Attributes:
        weights: The weights of the autoencoder. Encoder and decoder weights
            are saved independently, accessible through the *weights*
            dictionary. Individual weights are tensorflow variables.
        biases: The biases of the autoencoder. Biases are saved the same way
            as weights.
        frozen_layers: A list of integers specifying which layers should not
            be trained. Layers are numbered starting at 0.
        config: Configuration containing the parameters for the autoencoder.
        save_path: Path to save training checkpoints to.
        restore_path: Path to restore training checkpoints from.
            Note: The restore_path will be overriden by the `save_path` as
            soon as a checkpoint is saved. This ensures that the newest
            checkpoint is restored.
        layer_sizes: An array of layer sizes for the autoencoder. A symmetrical
            architecture is assumed so encoder and decoder layers do not have to
            be accounted for separetely.
        history_saver: Saves the history of the parameters of the autoencoder
    """
    def __init__(self, config, layer_sizes=None, restore_path=None):
        """ Initializes the autoencoder.

        Args:
            config: Configuration of the autoencoder.
            layer_sizes: List of integers specifying the size of the layers
                of the encoder part of the autoencoder. The decoder is
                generated symmetrically to the encoder.
            restore_path: Path from which the weights can be restored.
        """
        self.weights = {"encoder": [], "decoder": []}
        self.biases = {"encoder": [], "decoder": []}
        self.frozen_layers = []
        self._display_step = 1000
        self.config = config
        self.save_path = "./checkpoints/{}.ckpt".format(uuid4())
        self.restore_path = self.save_path
        self.layer_sizes = layer_sizes
        self.history_saver = HistorySaver()

        if self.layer_sizes != None:
            self.initialize_layers()

        if restore_path != None:
            self.restore_path = restore_path

    def initialize_layers(self):
        """ Initializes the weights and biases.

            Initializes the weights and biases of the autoencoder by creating
            layers from the list in *layer_sizes*.
        """
        for i in range(len(self.layer_sizes) - 1):
            source = self.layer_sizes[i]
            target = self.layer_sizes[i + 1]

            self._initialize_layer(source, target)

    def append_layer(self, layer_size):
        """ Adds an encoder and decoder layer to the middle of the autoencoder.

        Note:
            There has to be at least one layer in the autoencoder before
            calling this method.

        Args:
            layer_size: Size of the new layer.
        """

        # Get size of the currently last layer
        source = self.weights["encoder"][-1].shape[1].value
        target = layer_size

        self._initialize_layer(source, target)
        self.layer_sizes.append(layer_size)

        # Also append a layer to the configuration
        self.config.append_layer()

    def _initialize_layer(self, source_size, target_size):
        """ Creates a new layer from *source_size* to *target_size*.

        Creates a tensorflow variables with the shape defined by
        *source_size* and *target_size*. The variable is initialized with
        random values from the normal distribution. An encoder and decoder
        layer are generated with their biases and added to the autoencoder.

        Args:
            source_size: Size of the source layer.
            target_size: Size of the target layer.
        """
        self.weights["encoder"].append(tf.Variable(tf.random_normal([source_size, target_size])))
        self.weights["decoder"].insert(0, tf.Variable(tf.random_normal([target_size, source_size])))

        self.biases["encoder"].append(tf.Variable(tf.random_normal([target_size])))
        self.biases["decoder"].insert(0, tf.Variable(tf.random_normal([source_size])))

    def freeze_layer(self, layer):
        """ Freezes a layer of the autoencoder.

        The layer is frozen in the encoder and the decoder.

        Args:
            layer: Integer value specifying the layer to be frozen. Note that the first layer
            is specified by 0.
        """
        if layer not in self.frozen_layers:
            self.frozen_layers.append(layer)

    def unfreeze_layer(self, layer):
        """ Unfreezes a layer of the autoencoder.

        The layer is unfrozen in the encoder and the decoder.

        Args:
            layer: Integer value specifying the layer to be frozen. Note that the first layer
            is specified by 0.
        """
        if layer in self.frozen_layers:
            self.frozen_layers.remove(layer)

    def freeze_all_layers(self):
        """ Freezes all layers of the autoencoder. """
        self.frozen_layers = list(range(len(self.weights["encoder"])))

    def unfreeze_all_layers(self):
        """ Unfreezes all layers of the autoencoder. """
        self.frozen_layers.clear()

    def get_trainable_layers(self):
        """ Returns all trainable layers.

        Trainable layers are all layers that are not excluded by the
        *frozen_layers* list.
        """
        trainable_layers = []
        for i in range(len(self.weights["encoder"])):
            if i not in self.frozen_layers:
                trainable_layers.append(self.weights["encoder"][i])
                trainable_layers.append(self.weights["decoder"][i])

        return trainable_layers

    def encode(self, input_data, training=False):
        """ Encodes the input_data.

        Args:
            input_data: Vector of data to be encoded.
            training: If training is false, dropout is not used

        Returns:
            The vector of encoded data.
        """
        return self.forward_pass(input_data, self.weights["encoder"], self.biases["encoder"],
                                 self.config.activations["encoder"],
                                 self.config.dropout_rates["encoder"],
                                 training=training, linear_output=False)

    def decode(self, input_data, training=False):
        """ Decodes the input_data.

        Args:
            input_data: Vector of data to be decoded.
            training: If training is false, dropout is not used

        Returns:
            The vector of decoded data.
        """
        return self.forward_pass(input_data, self.weights["decoder"], self.biases["decoder"],
                                 self.config.activations["decoder"],
                                 self.config.dropout_rates["decoder"],
                                 training=training, linear_output=False)

    def full_pass(self, input_data, training=False):
        """ Encodes and decodes the input_data.

        Args:
            input_data: Vector of data to be encoded and decoded.
            training: If training is false, dropout is not used

        Returns:
            The vector of encoded and decoded data.
        """
        return self.decode(self.encode(input_data, training), training)

    def forward_pass(self, input_data, weights, biases, activations, dropout_rates,
                     training=False, linear_output=False):
        """ Passes input_data forward through the layers.

        The layers are defined by the *weights* and *biases*. In each layer
        the activation function specified in the *activations* list is used.
        A linear output can be used instead of the last activation function.

        Note:
            *weights*, *biases* and *activations* have to be in corresponding order.

        Args:
            input_data: Vector of data to be passed through the layers.
            weights: The weights to be used for the forward pass.
            biases: The biases to be used for the forward pass.
            activations: The activation functions for the forward pass.
            dropout_rates: Dropout rates for the forward pass
            training: If training is false, dropout is not used
            linear_output: If true, no activation function will be used on the output layer

        Returns:
            Vector of data after passing through the layers.
        """
        result = input_data

        # Seperate the last layer for possible linear output
        # Use dropout on every layer except on the output
        for index, (weight, bias) in enumerate(zip(weights[:-1], biases[:-1])):
            result = tf.layers.dropout(result, rate=dropout_rates[index], training=training)
            result = activations[index](tf.add(tf.matmul(result, weight), bias))

        result = tf.layers.dropout(result, rate=dropout_rates[-1], training=training)
        result = tf.add(tf.matmul(result, weights[-1]), biases[-1])
        if not linear_output:
            result = activations[-1](result)

        return result

    def _saver(self, num_layers=None):
        """ Creates a dictionary with the weigths to save or restore
        If *num_layers* is None all layers will be used

        Args:
            num_layers: Number of layers that should be restored

        Returns:
            A tensorflow saver that saves the number of layers specified.
        """
        save_dict = {}

        if num_layers is None:
            num_layers = len(self.weights["encoder"])

        for i in range(num_layers):
            save_dict["enc_w_{}".format(i)] = self.weights["encoder"][i]
            save_dict["dec_w_{}".format(i)] = self.weights["decoder"][-i-1]
            save_dict["enc_b_{}".format(i)] = self.biases["encoder"][i]
            save_dict["dec_b_{}".format(i)] = self.biases["decoder"][-i-1]

        return tf.train.Saver(var_list=save_dict, max_to_keep=0, write_version=tf.train.SaverDef.V2)

    @time_me
    def train(self, data, restore_layers=0, save=True):
        """ Trains the autoencoder using its configuration.

        Args:
            data: A fucntion that provides the input data for the network.
            restore_layers: Number of layers to restore prior to training.
            save: Set to true to save the training result, or false to discard it.

        Returns:
            Loss value (training score) of the autoencoder after training.
        """
        # Input to the network
        input_data = tf.placeholder("float", [None, self.layer_sizes[0]])

        # Loss function
        loss = tf.reduce_mean(tf.pow(input_data - self.full_pass(input_data, True), 2))
        loss_value = float("inf")

        # Optimizer
        optimizer = tf.train.RMSPropOptimizer(
            self.config.learning_rate, self.config.momentum).minimize(
                loss, var_list=self.get_trainable_layers())

        init = tf.global_variables_initializer()

        # Start training
        with tf.Session() as sess:
            sess.run(init)

            if restore_layers != 0:
                self._saver(restore_layers).restore(sess, self.restore_path)

            for i in range(self.config.num_steps):
                # Get the next batch (discard label because it is not needed)
                batch_x, _ = data.train.next_batch(self.config.batch_size)

                _, loss_value = sess.run([optimizer, loss], feed_dict={input_data: batch_x})

                # if i % self._display_step == 0:
                #     print("Step {}: Minibatch loss Loss: {}".format(i, loss_value))

            if save:
                save_path = self._saver().save(sess, self.save_path)
                # print("Weights and biases saved to: " + save_path)
                # If the results are saved the ae should restore from the new results
                self.restore_path = self.save_path

        return loss_value

    def run_encode_session(self, data):
        """ Runs a session to encode the *data*

        A tensorflow session is run encoding the data given in *data*.

        Args:
            data: A list of input vectors that should be encoded.
        """
        # Initialize the variables
        input_data = tf.placeholder("float", [None, self.layer_sizes[0]])
        loss = tf.reduce_mean(tf.pow(input_data - self.full_pass(input_data, False), 2))
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            self._saver().restore(sess, self.restore_path)

            # Encode and decode the digit image
            prediction, loss_value = sess.run([self.full_pass(input_data, False), loss],
                                              feed_dict={input_data: data})

        return prediction, loss_value

    def reconstruct_images(self, data):
        """ Plots reconstruction comparison of original and decoded images.

        Images are selected from the test set of the data. These images are
        encoded and decoded by the autoencoder and then compared to the
        original images in a plot.

        Args:
            data: A fucntion that provides the input data for the network.
        """
         # Encode and decode images from test set and visualize their reconstruction.
        width = 5
        height = 3
        num_images = 4
        canvas_orig = np.empty((28 * height, 28 * width))
        canvas_recon = np.empty((28 * height, 28 * width))

        input_data = tf.placeholder("float", [None, self.layer_sizes[0]])

        # Initialize the variables
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            self._saver().restore(sess, self.restore_path)

            for i in range(height):
                # MNIST test set
                batch_x, _ = data.test.next_batch(width)
                # Encode and decode the digit image
                prediction = sess.run(self.full_pass(input_data, False),
                                      feed_dict={input_data: batch_x})

                # Calculate original images
                for j in range(width):
                    canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                        batch_x[j].reshape([28, 28])

                # Calculate reconstructed images
                for j in range(width):
                    canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                        prediction[j].reshape([28, 28])

            plot_comparison(canvas_orig, canvas_recon, self.restore_path.split(".")[0] + ".png")

    def save_history(self):
        """ Saves the history to the same path as the checkpoints """
        history_path = self.save_path.split(".")[:-1]
        history_path.append("pickle")
        history_path = ".".join(history_path)

        self.history_saver.serialize(history_path)
