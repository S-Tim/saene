""" Simple test to see if tensorflow is working """

if __name__ == "__main__":
    import tensorflow as tf
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    print(sess.run(hello))
