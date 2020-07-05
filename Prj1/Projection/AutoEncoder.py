import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import sys
sys.path.append("..")
from processData import loadDataDivided
import SVMmodel

X_train, X_test, y_train, y_test = loadDataDivided(ifSubDir=True)
X_train_size = X_train.shape[0]
lr = 0.001
epochs = 1000
batch_size = 256
display_step = 1
n_input = 2048
weights = None
biases = None
data_pointer = 0

def getNextBatch(batch_size):
    global data_pointer
    if data_pointer + batch_size <= X_train_size:
        data_pointer += batch_size
        return X_train[data_pointer-batch_size:data_pointer], y_train[data_pointer-batch_size:data_pointer]
    else:
        return X_train[data_pointer:], y_train[data_pointer:]

def getHiddenLayerSettings(n_output):
    if n_output == 1024:
        return [1024]
    elif n_output == 512:
        return [1024, 512]
    elif n_output == 256:
        return [1024, 512, 256]
    elif n_output == 128:
        return [1024, 512, 256, 128]
    elif n_output == 64:
        return [1024, 512, 256, 128, 64]
    elif n_output == 2:
        return [1024, 512, 256, 128, 64, 16, 2]
    else:
        return []

def encoder(x, n_output):
    hidden_layers = getHiddenLayerSettings(n_output)
    layer = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    for i in range(2, len(hidden_layers)+1):
        layer = tf.nn.sigmoid(tf.add(tf.matmul(layer, weights['encoder_h'+str(i)]), biases['encoder_b'+str(i)]))
    return layer

def decoder(x, n_output):
    hidden_layers = getHiddenLayerSettings(n_output)
    layer = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    for i in range(2, len(hidden_layers)+1):
        layer = tf.nn.sigmoid(tf.add(tf.matmul(layer, weights['decoder_h'+str(i)]), biases['decoder_b'+str(i)]))
    return layer

def AutoEncoder(n_output):
    tf.compat.v1.disable_eager_execution()
    X = tf.compat.v1.placeholder("float", [None, n_input])
    global weights
    weights = {}
    global biases 
    biases = {}
    hidden_layers = getHiddenLayerSettings(n_output)
    for i in range(len(hidden_layers)):
        if i == 0:
            weights['encoder_h1'] = tf.Variable(tf.random.truncated_normal([n_input, hidden_layers[0]],))
        else:
            weights['encoder_h'+str(i+1)] = tf.Variable(tf.random.truncated_normal([hidden_layers[i-1], hidden_layers[i]],))
        biases['encoder_b'+str(i+1)] = tf.Variable(tf.random.normal([hidden_layers[i]]))
    
    reverseList = list(reversed(hidden_layers))
    for i in range(len(reverseList)):
        if i != len(reverseList) - 1:
            weights['decoder_h'+str(i+1)] = tf.Variable(tf.random.truncated_normal([reverseList[i], reverseList[i+1]],))
            biases['decoder_b'+str(i+1)] = tf.Variable(tf.random.normal([reverseList[i+1]]))
        else:
            weights['decoder_h'+str(i+1)] = tf.Variable(tf.random.truncated_normal([reverseList[i], n_input],))
            biases['decoder_b'+str(i+1)] = tf.Variable(tf.random.normal([n_input]))
    
    encoder_op = encoder(X, n_output)
    decoder_op = decoder(encoder_op, n_output)

    y_pred = decoder_op
    y_true = X

    cost = tf.compat.v1.reduce_mean(tf.square(y_true - y_pred))
    optimizer = tf.compat.v1.train.AdamOptimizer(lr).minimize(cost)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        total_batch = int(X_train_size / batch_size)

        for epoch in range(epochs):
            global data_pointer
            data_pointer = 0
            for i in range(total_batch):
                X_batch, y_batch = getNextBatch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={X : X_batch})
            if epoch % display_step == 0:
                print("Epoch: %d, Cost: %f"%(epoch+1, c))
        X_train_proj = sess.run(encoder_op, feed_dict={X : X_train})
        X_test_proj = sess.run(encoder_op, feed_dict={X : X_test})

    return X_train_proj, X_test_proj

def runAE(comp_range):
    rbf_scores = []
    linear_scores = []
    for n_comp in comp_range:
        print("\nn_comp=%d\n"%(n_comp))
        data_pointer = 0
        X_train_proj, X_test_proj = AutoEncoder(n_comp)
        if n_comp == 2:
            np.save('X_train_proj_2d_AE', X_train_proj)
            np.save('X_test_proj_2d_AE', X_test_proj)
        score_rbf = SVMmodel.runSVM(X_train_proj, X_test_proj, y_train, y_test, SVMmodel.getBestParam('rbf'), 'rbf')
        rbf_scores.append(score_rbf.mean())
        score_linear = SVMmodel.runSVM(X_train_proj, X_test_proj, y_train, y_test, SVMmodel.getBestParam('linear'), 'linear')
        linear_scores.append(score_linear.mean())
    return rbf_scores, linear_scores

def draw(comp_range, scores, kernel):
    bestIdx = np.argmax(scores)
    bestNComp = comp_range[bestIdx]
    bestAcc = scores[bestIdx]
    with open('res_AE_' + kernel + '.txt', 'w') as f:
        for i in range(len(comp_range)):
            f.write(kernel + ": n_comp = %f, acc = %f\n"%(comp_range[i], scores[i]))
        f.write(kernel + ": Best n_comp = %f\n"%(bestNComp))
        f.write(kernel + ": acc = %f\n"%(bestAcc))

    plt.figure()
    plt.plot(comp_range, scores, 'bo-', linewidth=2)
    plt.title('AE with SVM ' + kernel + ' kernel')
    plt.xlabel('n_components')
    plt.ylabel('Accuracy')
    plt.savefig('AE_' + kernel + '.jpg')

def main():
    comp_range = [2, 64, 128, 256, 512, 1024]
    rbf_scores, linear_scores = runAE(comp_range)
    draw(comp_range, rbf_scores, 'rbf')
    draw(comp_range, linear_scores, 'linear')

if __name__ == '__main__':
    main()



