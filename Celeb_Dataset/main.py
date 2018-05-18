
# coding: utf-8

# In[ ]:

print("UBitName = satvinde")
print("personNumber = 50248888")


# In[ ]:

import numpy as np,  glob as g, os as os, random
from scipy import misc
from PIL import Image
import tensorflow as tf


# In[ ]:

def resize_images(width, height, input_size):
    dir_name = "resized_"+str(width)+"_"+str(height)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    for filename in g.glob(os.path.join("img_align_celeba/", '*.jpg'))[0:input_size]:
        img=Image.open(filename.replace("\\","/"))
        img=img.resize((width,height), Image.NEAREST)
        img.save(filename.replace("img_align_celeba","resized_"+str(width)+"_"+str(height)))
    print("Resizing celeb images completed")    


# In[ ]:

def read_resized_images(width, height, input_size):
    temp=[]
    for i in range(1,input_size+1):
        img=misc.imread("resized_"+str(width)+"_"+str(height)+"/"+str(i).rjust(6,'0')+".jpg",flatten=1)
        img=np.ravel(img)
        temp.append(img)
    print("Reading resized celeb images completed")
    return np.array(temp)


# In[ ]:

def flip_images(width, height, input_size):
    dir_name = "flipped_"+str(width)+"_"+str(height)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    for filename in g.glob(os.path.join("resized_"+str(width)+"_"+str(height)+"/", '*.jpg'))[0:input_size]:
        img=Image.open(filename.replace("\\","/"))
        img=img.transpose(Image.FLIP_LEFT_RIGHT)
        img.save(filename.replace("resized","flipped"))    
    print("Flipping images completed")


# In[ ]:

def read_flipped_images(width, height, input_size):
    temp=[]
    for i in range(1,input_size+1):
        img=misc.imread("flipped_"+str(width)+"_"+str(height)+"/"+str(i).rjust(6,'0')+".jpg",flatten=1)
        img=np.ravel(img)
        temp.append(img)
    print("Reading flipped celeb images completed")
    return np.array(temp)


# In[ ]:

def get_augmented_data(width, height, input_size):
    resize_images(width, height, input_size)
    temp_img = read_resized_images(width, height, input_size)
    flip_images(width, height, input_size)
    temp_rotated_img = read_flipped_images(width, height, input_size)
    data=np.concatenate((temp_img,temp_rotated_img),axis=0)
    return data


# In[ ]:

def read_labels(input_size):
    temp=np.genfromtxt("list_attr_celeba.txt", delimiter=" ", skip_header=2, max_rows=input_size)
    temp=temp[0:,16]
    temp=np.reshape(temp,(-1,1))
    labels=[]
    for ele in temp:
        if ele == 1:
            labels.append([1,0])
        else:
            labels.append([0,1])
    print("Reading labels completed")
    return np.array(labels)    


# In[ ]:

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')


# In[ ]:

def train_and_get_accuracy(width, height, step_size, batch_size, no_of_neurons, prob, train_data, train_labels, test_data, test_labels):
    x = tf.placeholder(tf.float32, [None, width*height])
    y_ = tf.placeholder(tf.float32, [None, 2])

    W = tf.Variable(tf.zeros([width*height, 2]))
    b = tf.Variable(tf.zeros([1]))

    y = tf.matmul(x,W) + b

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, height, width, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([int(height/4) * int(width/4) * 64, no_of_neurons])
    b_fc1 = bias_variable([no_of_neurons])
    h_pool2_flat = tf.reshape(h_pool2, [-1, int(height/4) * int(width/4) * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([no_of_neurons, 2])
    b_fc2 = bias_variable([2])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(step_size):
            arr = random.sample(range(0,len(train_data)-1),batch_size)
            x_batch = []
            y_batch = []
            for ele in arr:
                x_batch.append(train_data[ele])
                y_batch.append(train_labels[ele])
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: x_batch, y_: y_batch, keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={x: x_batch, y_: y_batch, keep_prob: prob})
        print('Accuracy with Dataset: %g' % accuracy.eval(feed_dict={x: test_data, y_: test_labels, keep_prob: 1.0}))


# In[ ]:

print("\n\033[1mTraining with lower resolution images\033[0m")


# In[ ]:

input_size=5000
width=24
height=28


# In[ ]:

#resizing celeb images to a lower resolution
resize_images(width, height, input_size)


# In[ ]:

#reading resized celeb images
celeb_image_data=read_resized_images(width, height, input_size)


# In[ ]:

#print(celeb_image_data.shape)


# In[ ]:

#confirming image is proper
#temp=Image.fromarray(celeb_image_data[0].reshape(height,width))
#temp.show()


# In[ ]:

#reading labels
celeb_image_labels=read_labels(input_size)


# In[ ]:

#print(celeb_image_labels.shape)


# In[ ]:

#confirming correct labels
#image 154 has eyeGlasses so output should be 1,0
#print(celeb_image_labels[153])
#image 155 doesn't have eyeGlasses so output should be 0,1
#print(celeb_image_labels[154])


# In[ ]:

#creating training and testing sets
celeb_image_train_data = celeb_image_data[0:int(len(celeb_image_data)*0.8)]
celeb_image_train_labels = celeb_image_labels[0:int(len(celeb_image_labels)*0.8)]

celeb_image_test_data = celeb_image_data[int(len(celeb_image_data)*0.8):int(len(celeb_image_data))]
celeb_image_test_labels = celeb_image_labels[int(len(celeb_image_labels)*0.8):int(len(celeb_image_labels))]


# In[ ]:

print("Results with 100 neurons and 0.1 keep_prob")
train_and_get_accuracy(width, height, 1500, 100, 100, 0.1, celeb_image_train_data, celeb_image_train_labels, celeb_image_test_data, celeb_image_test_labels)


# In[ ]:

print("Results with 1024 neurons and 0.5 keep_prob")
train_and_get_accuracy(width, height, 1500, 100, 1024, 0.5, celeb_image_train_data, celeb_image_train_labels, celeb_image_test_data, celeb_image_test_labels)


# In[ ]:

print("\n\033[1mTraining with higher resolution images\033[0m")


# In[ ]:

input_size=5000
width=48
height=56


# In[ ]:

#resizing original celeb images to a resolution higher than previous
resize_images(width, height, input_size)


# In[ ]:

#reading resized celeb images
celeb_image_data=read_resized_images(width, height, input_size)


# In[ ]:

#print(celeb_image_data.shape)


# In[ ]:

#confirming image is proper
#temp=Image.fromarray(celeb_image_data[0].reshape(height,width))
#temp.show()


# In[ ]:

#reading labels
celeb_image_labels=read_labels(input_size)


# In[ ]:

#print(celeb_image_labels.shape)


# In[ ]:

#confirming correct labels
#image 154 has eyeGlasses so output should be 1,0
#print(celeb_image_labels[153])
#image 155 doesn't have eyeGlasses so output should be 0,1
#print(celeb_image_labels[154])


# In[ ]:

#creating training and testing sets
celeb_image_train_data = celeb_image_data[0:int(len(celeb_image_data)*0.8)]
celeb_image_train_labels = celeb_image_labels[0:int(len(celeb_image_labels)*0.8)]

celeb_image_test_data = celeb_image_data[int(len(celeb_image_data)*0.8):int(len(celeb_image_data))]
celeb_image_test_labels = celeb_image_labels[int(len(celeb_image_labels)*0.8):int(len(celeb_image_labels))]


# In[ ]:

train_and_get_accuracy(width, height, 1500, 100, 1024, 0.5, celeb_image_train_data, celeb_image_train_labels, celeb_image_test_data, celeb_image_test_labels)


# In[ ]:

print("\n\033[1mTraining with higher resolution images and larger training set\033[0m")


# In[ ]:

input_size=20000
width=48
height=56


# In[ ]:

#resizing original celeb images to a resolution higher than previous
resize_images(width, height, input_size)


# In[ ]:

#reading resized celeb images
celeb_image_data=read_resized_images(width, height, input_size)


# In[ ]:

#print(celeb_image_data.shape)


# In[ ]:

#confirming image is proper
#temp=Image.fromarray(celeb_image_data[0].reshape(height,width))
#temp.show()


# In[ ]:

#reading labels
celeb_image_labels=read_labels(input_size)


# In[ ]:

#print(celeb_image_labels.shape)


# In[ ]:

#confirming correct labels
#image 154 has eyeGlasses so output should be 1,0
#print(celeb_image_labels[153])
#image 155 doesn't have eyeGlasses so output should be 0,1
#print(celeb_image_labels[154])


# In[ ]:

#creating training and testing sets
celeb_image_train_data = celeb_image_data[0:int(len(celeb_image_data)*0.8)]
celeb_image_train_labels = celeb_image_labels[0:int(len(celeb_image_labels)*0.8)]

celeb_image_test_data = celeb_image_data[int(len(celeb_image_data)*0.8):int(len(celeb_image_data))]
celeb_image_test_labels = celeb_image_labels[int(len(celeb_image_labels)*0.8):int(len(celeb_image_labels))]


# In[ ]:

train_and_get_accuracy(width, height, 1500, 100, 1024, 0.5, celeb_image_train_data, celeb_image_train_labels, celeb_image_test_data, celeb_image_test_labels)


# In[ ]:

print("\n\033[1mTraining with higher resolution images and larger training set using data augmentation\033[0m")


# In[ ]:

input_size=20000
width=48
height=56


# In[ ]:

celeb_image_data=get_augmented_data(width, height, input_size)


# In[ ]:

#print(celeb_image_data.shape)


# In[ ]:

#confirming image is proper
#temp=Image.fromarray(celeb_image_data[20000].reshape(height,width))
#temp.show()


# In[ ]:

labels=read_labels(input_size)
celeb_image_labels=np.concatenate((labels,labels),axis=0)


# In[ ]:

#print(celeb_image_labels.shape)


# In[ ]:

#confirming correct labels
#image 154 has eyeGlasses so output should be 1,0
#print(celeb_image_labels[153])
#print(celeb_image_labels[20153])
#image 155 doesn't have eyeGlasses so output should be 0,1
#print(celeb_image_labels[154])


# In[ ]:

#creating training and testing sets
celeb_image_train_data = celeb_image_data[int(len(celeb_image_data)*0.2):int(len(celeb_image_data))]
celeb_image_train_labels = celeb_image_labels[int(len(celeb_image_labels)*0.2):int(len(celeb_image_labels))]

celeb_image_test_data = celeb_image_data[0:int(len(celeb_image_data)*0.2)]
celeb_image_test_labels = celeb_image_labels[0:int(len(celeb_image_labels)*0.2)]


# In[ ]:

train_and_get_accuracy(width, height, 1500, 100, 1024, 0.5, celeb_image_train_data, celeb_image_train_labels, celeb_image_test_data, celeb_image_test_labels)

