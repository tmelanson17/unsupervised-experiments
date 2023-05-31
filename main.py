import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

from torch.utils.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data

LOG_DIR = 'minimalsample'
NAME_TO_VISUALISE_VARIABLE = "mnistembedding"
TO_EMBED_COUNT = 500
SIZE=[224,224]



mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
batch_xs, batch_ys = mnist.train.next_batch(TO_EMBED_COUNT)

embedding_var = tf.Variable(batch_xs, name=NAME_TO_VISUALISE_VARIABLE)
summary_writer = tf.summary.FileWriter(LOG_DIR)

def create_embeddings(root_dir, dim):
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name

    # Specify where you find the metadata
    embedding.metadata_path = path_for_mnist_metadata #'metadata.tsv'

    # Create custom path
    path_for_mnist_sprites =  os.path.join(root_dir,'viz.png')
    path_for_mnist_metadata =  os.path.join(root_dir,'metadata.tsv')

    # Specify where you find the sprite (we will create this later)
    embedding.sprite.image_path = path_for_mnist_sprites 
    embedding.sprite.single_image_dim.extend(dim)
    return config

config = create_embeddings(LOG_DIR, SIZE)

# Say that you want to visualise the embeddings
projector.visualize_embeddings(summary_writer, config)

def create_sprite_image(images):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))


    spriteimage = np.ones((img_h * n_plots ,img_w * n_plots ))

    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h,
                  j * img_w:(j + 1) * img_w] = this_img

    return spriteimage

def vector_to_matrix(mnist_digits):
    """Reshapes normal mnist digit (batch,SIZE**2) to matrix (batch,28,28)"""
    return np.reshape(mnist_digits,(-1,SIZE[0],SIZE[1]))

def invert_grayscale(mnist_digits):
    """ Makes black white, and white black """
    return 1-mnist_digits

