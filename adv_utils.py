import tensorflow as tf
from keras.backend import categorical_crossentropy
from keras import backend as K
import numpy as np
import keras


def batch_eval(sess, tf_inputs, tf_outputs, numpy_inputs):
    """
    A helper function that computes a tensor on numpy inputs by batches.
    """
    n = len(numpy_inputs)
    assert n > 0
    assert n == len(tf_inputs)
    m = numpy_inputs[0].shape[0]
    for i in xrange(1, n):
        assert numpy_inputs[i].shape[0] == m
    out = []
    for _ in tf_outputs:
        out.append([])
    with sess.as_default():
        for start in xrange(0, m, FLAGS.batch_size):
            batch = start // FLAGS.batch_size
            if batch % 100 == 0 and batch > 0:
                print("Batch " + str(batch))

            # Compute batch start and end indices
            start = batch * FLAGS.batch_size
            end = start + FLAGS.batch_size
            numpy_input_batches = [numpy_input[start:end] for numpy_input in numpy_inputs]
            cur_batch_size = numpy_input_batches[0].shape[0]
            assert cur_batch_size <= FLAGS.batch_size
            for e in numpy_input_batches:
                assert e.shape[0] == cur_batch_size

            feed_dict = dict(zip(tf_inputs, numpy_input_batches))
            feed_dict[keras.backend.learning_phase()] = 0
            numpy_output_batches = sess.run(tf_outputs, feed_dict=feed_dict)
            for e in numpy_output_batches:
                assert e.shape[0] == cur_batch_size, e.shape
            for out_elem, numpy_output_batch in zip(out, numpy_output_batches):
                out_elem.append(numpy_output_batch)

    out = map(lambda x: np.concatenate(x, axis=0), out)
    for e in out:
        assert e.shape[0] == m, e.shape
    return out


def run_batch_generator(model=None, 
                        generator=None, 
                        inputs=None, 
                        outputs = None, 
                        learning_phase = None,
                        sess=None, nbsamples = None):
    '''
    Executes any computation graph on batches obtained from generator 
    Returned the specified output
    '''
    out = []
    for _ in outputs:
        out.append([])
    with sess.as_default():
        #time to run the session!!
        samples_seen = 0
        while samples_seen < nbsamples:
            X,_ = generator.__next__()
            samples_seen+=X.shape[0]
            feed_dict = dict()
            feed_dict[inputs] = X
            feed_dict[K.learning_phase()] = learning_phase
            batch_out = sess.run(outputs,feed_dict = feed_dict)
            for out_elem, batch_out_ele in zip(out, batch_out):
                out_elem.append(batch_out_ele)
        print('Total samples seen', samples_seen)
    out = list(map(lambda x: np.concatenate(x, axis=0), out))
    return out

def fgsm_generator(model=None, generator=None, nbsamples=None, epsilon=None, savedir=None, sess=None):
    '''
    creates and executes the adv_x ops on image batches obtained from generator
    '''
    adv_x,x,predictions = fgsm_graph(model, eps=epsilon)
     
    out = run_batch_generator(model=model, 
                        generator=generator, 
                        inputs=x, 
                        outputs = [adv_x],
                        learning_phase = 0,
                        sess=sess, nbsamples = nbsamples)
    print(out[0].shape)
    
    ''' TODO: sabe the adv in a specified location; make a wrapper over this function'''
    return out[0]
    #X_test_adv, = batch_eval(sess, [x], [adv_x], [X_test])

def fgsm_graph(model=None, eps=None):
    '''
    Creates adv_x ops 
    '''

    #define a placeholder for input images
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    #define the computation graph
    predictions = model(x)

    #compute loss
    y = tf.to_float(tf.equal(predictions, tf.reduce_max(predictions, 1, keep_dims=True))) #compare with its max
    y = y / tf.reduce_sum(y, 1, keep_dims=True) #normalise
    loss = tf.reduce_mean(categorical_crossentropy(y, predictions))
    
    # Define gradient of loss wrt input
    grad, = tf.gradients(loss, x)

    # Take sign of gradient
    signed_grad = tf.sign(grad)

    # Multiply by constant epsilon
    scaled_signed_grad = eps * signed_grad

    # Add perturbation to original example to obtain adversarial example
    adv_x = tf.stop_gradient(x + scaled_signed_grad)


    return adv_x, x, grad


def stochastic_prediction(model=None, generator=None, nbsamples=None, num_feed_forwards=10, savedir=None, sess=None):
    '''
    Creates and executes ops for stochastic prediction
    '''
    
    #define a placeholder for input images
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    
    #define the computation graph
    #predictions = tf.concat(0,[[model(x)] for _ in range(num_feed_forwards)])
    predictions = model(x)

    '''
    pred_argmax = tf.argmax(predictions, 1, name="predictions")
    correct_predictions = tf.equal(pred_argmax, tf.argmax(y, 1))

    #self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
    learning_phase = 0
    outputs = [correct_predictions]
    out = []
    for _ in outputs:
        out.append([])
    with sess.as_default():
        #time to run the session!!
        samples_seen = 0
        while samples_seen < nbsamples:
            X,Y = generator.__next__()
            samples_seen+=X.shape[0]
            feed_dict = dict()
            feed_dict[x] = X
            feed_dict[y] = Y
            feed_dict[K.learning_phase()] = learning_phase
            batch_out = sess.run(outputs,feed_dict = feed_dict)
            for out_elem, batch_out_ele in zip(out, batch_out):
                out_elem.append(batch_out_ele)
    out = list(map(lambda x: np.concatenate(x, axis=0), out))
    ''' 
    # Define sympbolic for accuracy
    acc_value = keras.metrics.categorical_accuracy(y, predictions)

    # Init result var
    accuracy = 0.0

    with sess.as_default():
        # Compute number of batches

        samples_seen = 0
        while samples_seen < nbsamples:
            X,Y = generator.__next__()
            samples_seen+=X.shape[0]
            #feed_dict = dict()
            #feed_dict[x] = X
            #feed_dict[y] = Y
            #feed_dict[K.learning_phase()] = learning_phase

            # The last batch may be smaller than all others, so we need to
            # account for variable batch size here
            accuracy += X.shape[0] * acc_value.eval(feed_dict={x: X,
                                            y: Y,
                                            keras.backend.learning_phase(): 0})

        # Divide by number of examples to get final value
        accuracy /= nbsamples


  
    '''
    #execute the the ops in Tf sessions
    out = run_batch_generator(model=model, 
                        generator=generator, 
                        inputs=x, 
                        outputs = [predictions],
                        learning_phase = 1,
                        sess=sess, nbsamples = nbsamples)
    '''
    #compute accuracy from the scores obtained output
    print(accuracy)
    #return out
