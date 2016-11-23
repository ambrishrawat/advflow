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


def run_batch_generator(generator=None, 
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
    out = list(map(lambda x: np.concatenate(x, axis=0), out))
    return out

def fgsm_generator(model=None, generator=None, nbsamples=None, epsilon=None, sess=None):
    '''
    creates and executes the adv_x ops on image batches obtained from generator
    '''


    adv_x,x,predictions = fgsm_graph_away(model, eps=epsilon)
     
    out = run_batch_generator(generator=generator, 
                        inputs=x, 
                        outputs = [adv_x, predictions],
                        learning_phase = 0,
                        sess=sess, nbsamples = nbsamples)
    print(out[0].shape)
    return out[0], out[1]

def fgsm_generator_towards(model=None, 
        towards_labels = None, 
        generator=None, 
        nbsamples=None, 
        epsilon=None, 
        sess=None):

    '''
    creates and executes the adv_x ops on image batches obtained from generator
    '''

    adv_x,x,y = fgsm_graph_towards(model, eps=epsilon)
     
    outputs = [adv_x]
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
            feed_dict[x] = X
            feed_dict[y] = towards_labels
            feed_dict[K.learning_phase()] = 0
            batch_out = sess.run(outputs,feed_dict = feed_dict) 
            for out_elem, batch_out_ele in zip(out, batch_out):
                out_elem.append(batch_out_ele)

    out = list(map(lambda x: np.concatenate(x, axis=0), out))
    
    return out[0]


def fgsm_graph_towards(model=None, eps=None):
    '''
    Creates adv_x ops 
    '''

    #define a placeholder for input images
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    #define the computation graph
    predictions = model(x)

    ''' Loss for the predicted label '''

    #compute loss
    loss = tf.reduce_mean(categorical_crossentropy(y, predictions))
    
    # Define gradient of loss wrt input
    grad, = tf.gradients(loss, x)

    # Take sign of gradient
    signed_grad = tf.sign(grad)

    # Multiply by constant epsilon
    scaled_signed_grad = eps * signed_grad

    # Add perturbation to original example to obtain adversarial example
    adv_x = tf.stop_gradient(x + scaled_signed_grad)


    return adv_x, x, y

def fgsm_graph_away(model=None, eps=None):
    '''
    Creates adv_x ops 
    '''

    #define a placeholder for input images
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    #define the computation graph
    predictions = model(x)

    ''' Loss for the predicted label '''

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


    return adv_x, x, y


def stoch_grads(model=None, eps=None):
    '''
    Creates stoch_grad ops and executes them
    '''

    #define a placeholder for input images
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    #define the computation graph
    predictions = model(x)

    ''' Loss for the predicted label '''

    #compute loss
    y = tf.to_float(tf.equal(predictions, tf.reduce_max(predictions, 1, keep_dims=True))) #compare with its max
    y = y / tf.reduce_sum(y, 1, keep_dims=True) #normalise
    loss = tf.reduce_mean(categorical_crossentropy(y, predictions))
    
    # Define gradient of loss wrt input
    grad, = tf.gradients(loss, x)

    grads = []
    for _ in range(num_feed_forwards):
        out = run_batch_generator(generator=generator, 
                            inputs=x, 
                            outputs = [grad],
                            learning_phase = 1,
                            sess=sess, nbsamples = nbsamples)
        grads.append(out[0])

    return np.mean(grads,axis=0)
    


def mc_dropout_preds(model=None, 
        generator=None, 
        nbsamples=None, 
        num_feed_forwards=10, 
        sess=None):

    '''
    Creates and executes ops for stochastic prediction
    '''
    
    #define a placeholder for input images
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    
    #define the computation graph
    predictions = model(x) 

    stochs = []
    for _ in range(num_feed_forwards):
        out = run_batch_generator(generator=generator, 
                            inputs=x, 
                            outputs = [predictions],
                            learning_phase = 1,
                            sess=sess, nbsamples = nbsamples)
        stochs.append(out[0])

    return np.asarray(stochs)

def mc_dropout_eval_helper(labels=None,stoch_preds=None, sess=None):

    y = tf.placeholder(tf.float32, shape=(labels.shape[0],10))
    predictions = tf.placeholder(tf.float32, shape=(stoch_preds.shape[0],labels.shape[0],10))
    mc_approx = tf.reduce_mean(predictions,0)

    pred_argmax = tf.argmax(mc_approx, 1, name="predictions")
    correct_predictions = tf.equal(pred_argmax, tf.argmax(y, 1))
    accuracy = 0.0
    accuracy_batch = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
    learning_phase = 0
    with sess.as_default():
        #time to run the session!!
        feed_dict = dict()
        feed_dict[y] = labels
        feed_dict[predictions] = stoch_preds
        feed_dict[K.learning_phase()] = learning_phase
        batch_out = sess.run([accuracy_batch],feed_dict = feed_dict)
        accuracy+=batch_out[0]
    
    
    #compute accuracy from the scores obtained output
    return accuracy

def mc_dropout_eval(model=None, 
        generator=None, 
        nbsamples=None, 
        num_feed_forwards=10, 
        sess=None,
        labels=None):


    preds = mc_dropout_preds(model=model, 
        generator=generator, 
        nbsamples=nbsamples, 
        num_feed_forwards=num_feed_forwards, 
        sess=sess)

    return mc_dropout_eval_helper(labels=labels[0:nbsamples],stoch_preds=preds, sess=sess)


def mc_dropout_stats_helper(labels=None,stoch_preds=None, sess=None):

    y = tf.placeholder(tf.float32, shape=(labels.shape[0],10))
    predictions = tf.placeholder(tf.float32, shape=(stoch_preds.shape[0],labels.shape[0],10))
    mc_approx = tf.reduce_mean(predictions,0)
    #predictions = model(x)

    #multiply mc_approx by y to get (confidence for label y) 
    mean_along_y = tf.reduce_max(tf.mul(mc_approx,y), 1, keep_dims = True)

    #(variance instead of std now) std dev (a measure of uncertainity in confidence for label y))
    e_xx = tf.reduce_mean(tf.mul(predictions,predictions),0) 
    std_dev_along_y = tf.sub(e_xx,tf.mul(mc_approx,mc_approx))

    #variational ratio
    temp = tf.to_float(tf.equal(predictions, tf.reduce_max(predictions, 2, keep_dims=True))) #compare with its max
    temp = temp / tf.reduce_sum(temp, 2, keep_dims=True) #normalise
    temp = tf.reduce_sum(temp,0) #reduce sum across T feed forwards
    temp = tf.reduce_max(temp,1)
    
    outputs = [mean_along_y,std_dev_along_y, temp]

    with sess.as_default():
        #time to run the session!!
        feed_dict = dict()
        feed_dict[y] = labels
        feed_dict[predictions] = stoch_preds
        feed_dict[K.learning_phase()] = 0
        out = sess.run(outputs,feed_dict = feed_dict)
         
    return out


def mc_dropout_stats(model=None, 
        generator=None, 
        nbsamples=None, 
        num_feed_forwards=10, 
        sess=None,
        labels=None):

    preds = mc_dropout_preds(model=model, 
        generator=generator, 
        nbsamples=nbsamples, 
        num_feed_forwards=num_feed_forwards, 
        sess=sess)

    out = mc_dropout_stats_helper(labels=labels[0:nbsamples],stoch_preds=preds, sess=sess)

    mc_acc = mc_dropout_eval_helper(labels=labels[0:nbsamples],stoch_preds=preds, sess=sess)
    return preds, out[0], out[1], out[2], mc_acc




def std_dropout_stats(model=None, 
        generator=None, 
        nbsamples=None, 
        sess=None,
        labels=None):

    '''
    NOTE: tf.pack makes a copy of the same tf vairable and hence model(x) is not executed n times

    TODO: try by adding to collection
    '''

    #define a placeholder for input images
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    
    #define the computation graph
    predictions = model(x)

    #multiply mc_approx by y to get (confidence for label y) 
    mean_along_y = tf.reduce_max(tf.mul(predictions,y), 1, keep_dims = True)
   
    #accuracy with respect to y
    pred_argmax = tf.argmax(predictions, 1, name="predictions")
    correct_predictions = tf.equal(pred_argmax, tf.argmax(y, 1))
    accuracy = 0.0
    accuracy_batch = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
    
    learning_phase = 0
    outputs = [mean_along_y, predictions]
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
            batch_out = sess.run([accuracy_batch]+outputs,feed_dict = feed_dict)
            accuracy+=batch_out[0]*X.shape[0] 
            for out_elem, batch_out_ele in zip(out, batch_out[1:]):
                out_elem.append(batch_out_ele)

    out = list(map(lambda x: np.concatenate(x, axis=0), out))
    
    #compute accuracy from the scores obtained output
    accuracy/=nbsamples 
    return out[0], out[1], accuracy


def nearest_in_set(adv, X_train):
    ''' 
    Computed the min L2 - distance of each image in adv from X_train
    '''
    '''
    def dist_(x,y):
        return np.linalg.norm(x.flatten()-y.flatten())

    def dist_helper(x):
        return np.min([dist_(x,X_train[i]) for i in range(X_train.shape[0])])
    
    return np.asarray([dist_helper(adv[i]) for i in range(adv.shape[0])])
    '''
    c_Xtrain = X_train
    min_dis = np.sum(np.square(adv - c_Xtrain[0:adv.shape[0]]),axis=(1,2,3))
    for _ in range(X_train.shape[0]):
        c_Xtrain = np.roll(c_Xtrain,1,axis=0)
        curr_ = np.sum(np.square(adv - c_Xtrain[0:adv.shape[0]]),axis=(1,2,3))
        min_dis = map(lambda x,y : min(x,y), zip(min_dis,curr_))

    return min_dis

