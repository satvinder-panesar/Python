
# coding: utf-8

# In[1]:

print("UBitName = satvinde")
print("personNumber = 50248888")

# In[1]:

import sklearn.cluster as sc, numpy as np, math, os as os, glob as g
from scipy import misc
from PIL import Image

# In[59]:

def resize_usps_images():
    for i in range(0,10):
        for filename in g.glob(os.path.join("proj3_images/Numerals/"+str(i), '*.png')):
            img=Image.open(filename.replace("\\","/"))
            img=img.resize((28,28), Image.NEAREST)
            img.save(filename.replace("Numerals","Numerals_Resized"))   


# In[60]:

def read_resized_usps_images(usps_test_data,usps_test_1d_label,usps_test_2d_label):
    for i in range(0,10):
        for filename in g.glob(os.path.join("proj3_images/Numerals_Resized/"+str(i), '*.png')):
            temp=[]
            img=misc.imread(filename.replace("\\","/"),flatten=1)
            img[img==0]=1
            img[img==255]=0
            img=np.ravel(img)
            usps_test_data.append(img)
            if i==0:
                usps_test_2d_label.append([1,0,0,0,0,0,0,0,0,0])
                usps_test_1d_label.append(0)            
            elif i==1:
                usps_test_2d_label.append([0,1,0,0,0,0,0,0,0,0])
                usps_test_1d_label.append(1)
            elif i==2:
                usps_test_2d_label.append([0,0,1,0,0,0,0,0,0,0])
                usps_test_1d_label.append(2)
            elif i==3:
                usps_test_2d_label.append([0,0,0,1,0,0,0,0,0,0])
                usps_test_1d_label.append(3)
            elif i==4:
                usps_test_2d_label.append([0,0,0,0,1,0,0,0,0,0])
                usps_test_1d_label.append(4)
            elif i==5:
                usps_test_2d_label.append([0,0,0,0,0,1,0,0,0,0])
                usps_test_1d_label.append(5)
            elif i==6:
                usps_test_2d_label.append([0,0,0,0,0,0,1,0,0,0])
                usps_test_1d_label.append(6)
            elif i==7:
                usps_test_2d_label.append([0,0,0,0,0,0,0,1,0,0])
                usps_test_1d_label.append(7)
            elif i==8:
                usps_test_2d_label.append([0,0,0,0,0,0,0,0,1,0])
                usps_test_1d_label.append(8)
            elif i==9:
                usps_test_2d_label.append([0,0,0,0,0,0,0,0,0,1])
                usps_test_1d_label.append(9)
    usps_test_data=np.array(usps_test_data)
    usps_test_2d_label=np.array(usps_test_2d_label)
    usps_test_1d_label=np.array(usps_test_1d_label)
    #print(usps_test_data.shape)
    #shape of mnist_data (19999,784)
    #print(usps_test_2d_label.shape)
    #shape of usps_test_2d_label (19999,10)
    #print(usps_test_1d_label.shape)
    #shape of usps_test_1d_label (19999,)    


# In[61]:

def get_bias_function(classifier):
    return np.around(classifier.intercept_,decimals=2)


# In[62]:

def get_accuracy(classifier,data,target,dataset):
    print("Accuracy with "+dataset+" Dataset: ",classifier.score(data, target))    

# In[100]:

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# In[101]:

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

# In[63]:

#Resize USPS images to 28x28
resize_usps_images()
print("\nResizing USPS images completed") 


# In[64]:

#Reading all resized USPS images to generate data and label
usps_test_data=[]
usps_test_1d_label=[]
usps_test_2d_label=[]
read_resized_usps_images(usps_test_data,usps_test_1d_label,usps_test_2d_label)
print("\nReading resized USPS images completed")


# In[4]:

#confirm that correct image is stored
#temp=np.copy(usps_test_data[10000])
#temp[temp==0]=255
#temp[temp==1]=0
#temp=Image.fromarray(np.reshape(temp,(28,28)))
#temp.show()
#print(usps_test_2d_label[10000])
#print(usps_test_1d_label[10000])


# In[66]:

#prints the first entry in USPS test data
print("\nFirst scanned image of resized USPS data\n")
matrix=np.matrix(np.reshape(usps_test_data[0],(28,28)),dtype=int)
print(matrix)


# In[5]:

print("\n\033[1m*****Logistic regression*****\n     -------------------\033[0m")


# In[35]:

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
mnist.data[mnist.data>0]=1
#shape of MNIST data (70000, 784)
#print(mnist.data.shape)


# In[36]:

#Splitting MNIST data into train(80%) and test(20%) sets 
mnist_train_data=mnist.data[0:int(len(mnist.data)*0.8)]
mnist_train_target=mnist.target[0:int(len(mnist.target)*0.8)]
mnist_test_data=mnist.data[int(len(mnist.data)*0.8):int(len(mnist.data))]
mnist_test_target=mnist.target[int(len(mnist.target)*0.8):int(len(mnist.target))]


# In[37]:

#Disabling bias
lr=LogisticRegression(fit_intercept=False)


# In[38]:

#Training model without bias
lr.fit(mnist_train_data, mnist_train_target)


# In[ ]:
print("\n\033[1mValidation phase\033[0m");
print("\nResults with no bias:\n-------------------------")


# In[39]:

#Testing MNIST dataset
get_accuracy(lr,mnist_test_data,mnist_test_target,"MNIST")


# In[41]:

#Enabling bias
lr=LogisticRegression(fit_intercept=True)


# In[42]:

#Training model witht bias
lr.fit(mnist_train_data, mnist_train_target)


# In[43]:

#Bias added to the decision function
print("\nBias added to decision function\n",get_bias_function(lr))

# In[ ]:

print("\nResults with bias:\n----------------------")

# In[44]:

#Testing MNIST dataset
get_accuracy(lr,mnist_test_data,mnist_test_target,"MNIST")



# In[45]:

#Testing USPS dataset
print("\n\033[1mTesting phase\033[0m");
print("\nResults with tuned parameters:\n-------------------------")
get_accuracy(lr,usps_test_data,usps_test_1d_label,"USPS")


# In[13]:

print("\n\033[1m*****Single Layer Neural Network*****\n     ---------------------------\033[0m")


# In[50]:

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)


# In[51]:

y_ = tf.placeholder(tf.float32, [None, 10])


# In[52]:

#Formula for cross_entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))


# In[85]:

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)


# In[86]:

#Initialize session
sess = tf.InteractiveSession()

tf.global_variables_initializer().run()


# In[87]:

#Updating weights batch wise
for _ in range(500):
    batch_xs, batch_ys = mnist.train.next_batch(50)
    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})


# In[88]:

#Correct formula
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))


# In[89]:

#Accuracy formula
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


# In[90]:
print("\n\033[1mValidation phase\033[0m");
print("\nResults with learning rate=0.1, batch_size=50 and no_of_iterations=500\n---------------------------------------------------------------------------")


# In[91]:

#Testing with MNIST Data
print("Accuracy with MNIST Dataset: ",sess.run(accuracy, feed_dict={x:mnist.test.images, y_: mnist.test.labels}))

# In[93]:

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# In[94]:

#Initialize session
sess = tf.InteractiveSession()

tf.global_variables_initializer().run()


# In[95]:

#Updating weights batch wise
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})


# In[96]:

print("\nResults with learning rate=0.5, batch_size=100 and no_of_iterations=1000\n-----------------------------------------------------------------------------")


# In[97]:

#Testing with MNIST Data
print("Accuracy with MNIST Dataset: ",sess.run(accuracy, feed_dict={x:mnist.test.images, y_: mnist.test.labels}))


# In[98]:

#Testing with USPS Data
print("\n\033[1mTesting phase\033[0m");
print("\nResults with tuned parameters:\n-------------------------")
print("Accuracy with USPS Dataset: ",sess.run(accuracy, feed_dict={x:usps_test_data, y_: usps_test_2d_label}))


# In[99]:

print("\n\033[1m*****MultiLayer Convolutional Network*****\n     --------------------------------\033[0m")


# In[102]:

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# In[103]:

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# In[104]:

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# In[105]:

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# In[106]:

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


# In[107]:

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[108]:

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob:0.5})
    print('Accuracy with MNIST Dataset: %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    print('Accuracy with USPS Dataset: %g' % accuracy.eval(feed_dict={x: usps_test_data, y_: usps_test_2d_label, keep_prob: 1.0}))


# In[ ]:



