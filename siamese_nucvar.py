"""
Siamese neural network module.

original from Abram Spamers at https://github.com/aspamers/siamese/blob/master/siamese.py   commit: Latest commit fed9a92 on Aug 24, 2020
modified by Lex Flagel

"""

import random
import numpy as np

from keras.layers import Input
from keras.models import Model


class SiameseNetwork:
    """
    A simple and lightweight siamese neural network implementation.

    The SiameseNetwork class requires the base and head model to be defined via the constructor. The class exposes
    public methods that allow it to behave similarly to a regular Keras model by passing kwargs through to the
    underlying keras model object where possible. This allows Keras features like callbacks and metrics to be used.
    """
    def __init__(self, base_model, head_model):
        """
        Construct the siamese model class with the following structure.

        -------------------------------------------------------------------
        input1 -> base_model |
                             --> embedding --> head_model --> binary output
        input2 -> base_model |
        -------------------------------------------------------------------

        :param base_model: The embedding model.
        * Input shape must be equal to that of data.
        :param head_model: The discriminator model.
        * Input shape must be equal to that of embedding
        * Output shape must be equal to 1..
        """
        # Set essential parameters
        self.base_model = base_model
        self.head_model = head_model

        # Get input shape from base model
        self.input_shape = self.base_model.input_shape[1:]

        # Initialize siamese model
        self.siamese_model = None
        self.__initialize_siamese_model()

    def compile(self, *args, **kwargs):
        """
        Configures the model for training.

        Passes all arguments to the underlying Keras model compile function.
        """
        self.siamese_model.compile(*args, **kwargs)
        
    def summary(self):
        """
        Configures the model for training.

        Passes all arguments to the underlying Keras model compile function.
        """
        self.siamese_model.summary()
        
    def fit(self, *args, **kwargs):
        """
        Trains the model on data generated batch-by-batch using the siamese network generator function.

        Redirects arguments to the fit_generator function.
        """
        x_train = args[0]
        x_test = args[1]
        batch_size = kwargs.pop('batch_size')

        train_generator = self.__pair_generator(x_train,batch_size)
        train_steps = max(len(x_train) / batch_size, 1)
        test_generator = self.__pair_generator(x_test, batch_size)
        test_steps = max(len(x_test) / batch_size, 1)
        self.siamese_model.fit(train_generator,
                               steps_per_epoch=train_steps,
                               validation_data=test_generator,
                               validation_steps=test_steps, **kwargs)

    def load_weights(self, checkpoint_path):
        """
        Load siamese model weights. This also affects the reference to the base and head models.

        :param checkpoint_path: Path to the checkpoint file.
        """
        self.siamese_model.load_weights(checkpoint_path)

    def evaluate(self, *args, **kwargs):
        """
        Evaluate the siamese network with the same generator that is used to train it. Passes arguments through to the
        underlying Keras function so that callbacks etc can be used.

        Redirects arguments to the evaluate_generator function.

        :return: A tuple of scores
        """
        x = args[0]
#        y = args[1]
        batch_size = kwargs.pop('batch_size')

        generator = self.__pair_generator(x, batch_size)
        steps = len(x) / batch_size
        return self.siamese_model.evaluate_generator(generator, steps=steps, **kwargs)

    def __initialize_siamese_model(self):
        """
        Create the siamese model structure using the supplied base and head model.
        """
        input_a = Input(shape=self.input_shape)
        input_b = Input(shape=self.input_shape)
        #print(input_a)
        #print(input_b)
        processed_a = self.base_model(input_a)
        processed_b = self.base_model(input_b)

        head = self.head_model([processed_a, processed_b])
        self.siamese_model = Model([input_a, input_b], head)

    def __pair_generator(self, x, batch_size):
        """
        Creates a python generator that produces pairs from the original input data.
        :param x: Input data
        :param batch_size: The number of pair samples to create per batch.
        :return:
        """
        x_seg_site_D = {} #dict of lists of matrices with grouped by number of seg sites, with seg sites as key
        for i in x:
            seg_sites = sum(i.sum(axis=1) > 0)[0]
            #print(seg_sites)
            if seg_sites not in x_seg_site_D : x_seg_site_D[seg_sites] = []
            x_seg_site_D[seg_sites].append(i)
        for i in list(x_seg_site_D.keys()): #delete out any with less than 2, b/c there is no neg. pair possible
            if len(x_seg_site_D[i]) < 2: del(x_seg_site_D[i])
        
        #class_indices, num_classes = self.__get_class_indices(y)
        while True:
            pairs, labels = self.__create_pairs(x_seg_site_D, batch_size)

            # The siamese network expects two inputs and one output. Split the pairs into a list of inputs.
            yield [pairs[:, 0], pairs[:, 1]], labels

    def __create_pairs(self, x_seg_site_D, batch_size):
        """
        Create a numpy array of positive and negative pairs and their associated labels.

        :param x: Input data
        :param batch_size: The number of pair samples to create.
        :return: A tuple of (Numpy array of pairs, Numpy array of labels)
        """
        def indv_ord_augmentation(almt):
            q = almt.copy()
            np.random.shuffle(q.T)
            return q
        
        num_pairs = batch_size / 2
        positive_pairs, positive_labels = [],[] 
        negative_pairs, negative_labels = [],[]
        seg_site_list = list(x_seg_site_D.keys())
        for _ in range(int(num_pairs)):
            s = random.choice(seg_site_list)
            duo = random.choices(x_seg_site_D[s], k=2)
            positive_pair = [indv_ord_augmentation(duo[0]), indv_ord_augmentation(duo[0])]
            positive_label = [1.0]
            negative_pair = [indv_ord_augmentation(duo[0]), indv_ord_augmentation(duo[1])]
            negative_label = [0.0]
            positive_pairs.append(positive_pair)
            negative_pairs.append(negative_pair)
            positive_labels.append(positive_label)
            negative_labels.append(negative_label)

        return np.array(positive_pairs + negative_pairs), np.array(positive_labels + negative_labels)
    
    def save(self, filepath, *args, **kwargs):
        self.siamese_model.save(filepath, *args, **kwargs)
        self.base_model.save(filepath+'_base', *args, **kwargs)
        self.head_model.save(filepath+'_head', *args, **kwargs)
