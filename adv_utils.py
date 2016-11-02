import tensorflow as tf
from keras.backend import categorical_crossentropy
from keras import backend as K




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

def fgsm_generator(model=None, generator=None, nbsamples=None, epsilon=None, savedir=None, sess=None):
    '''
    Generates adversrial images for a trained model
    '''
    adv_x,x,grads = fgsm_graph(model, eps=epsilon)
   
    with sess.as_default(): 
        #time to run the session!!
        samples_seen = 0
        while samples_seen <= nbsamples:
            X,_ = generator.__next__() 
            samples_seen+=X.shape[0]
            feed_dict = dict()
            feed_dict[x] = X
            feed_dict[K.learning_phase()] = 1
            output = sess.run([adv_x,grads],feed_dict = feed_dict)

            adv_X = output[0]
            grads = output[1]
            print('Yoda '+str(X.shape[0])+ ' '+ str(samples_seen))
            print(sum(grads))
    #X_test_adv, = batch_eval(sess, [x], [adv_x], [X_test])

def fgsm_graph(model=None, eps=None):
    '''
    Generates adversrial images for a trained model
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
