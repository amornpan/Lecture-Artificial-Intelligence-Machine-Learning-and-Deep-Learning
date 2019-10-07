== Introduction to Computer Vision
So in that lesson, we just saw the basics of the new programming paradigm that comes with machine learning and deep learning, and how instead of like expressing rules in a programming language, we can start getting data and using labeled data to open up new scenarios like activity recognition. Then for a little bit of fun, we actually started doing our first piece of code. We built a super simple neural network that fit data like an x and y data onto a line but that was just "Hello, World". Right, Andrew? So fitting straight lines seems like the "Hello, world" most basic implementation learning algorithm. But one of the most amazing things about machine learning is that, that core of the idea of fitting the x and y relationship is what lets us do amazing things like, have computers look at the picture and do activity recognition, or look at the picture and tell us, is this a dress, or a pair of pants, or a pair of shoes really hard for humans, and amazing that computers can now use this to do these things as well. Right, like computer vision is a really hard problem to solve, right? Because you're saying like dress or shoes. It's like how would I write rules for that? How would I say, if this pixel then it's a shoe, if that pixel then its a dress. It's really hard to do, so the labeled samples are the right way to go. Yeah. One of the non-intuitive things about vision is that it's so easy for a person to look at you and say, you're wearing a shirt, it's so hard for a computer to figure it out. Because it's so easy for humans to recognize objects, it's almost difficult to understand why this is a complicated thing for a computer to do. What the computer has to do is look at all numbers, all the pixel brightness value, saying look at all of these numbers saying, these numbers correspond to a black shirt, and it's amazing that with machine and deep learning computers are getting really good at this. Right, so it's like with the code that we just used in the previous lesson as you mentioned, it provides a template for everything that we can do with deep learning by designing a Neural network in the layers to be able to recognize patterns like this. So maybe we can do that with clothes recognition today. What do you think? Yeah. So in the next video, you'll learn how to write code to take this paradigm you've already saw in a previous video, but to now apply it to recognizing clothes from labeled data. Please go on to the next video.

== An Introduction to computer vision
In the previous lesson, you learned what the machine learning paradigm is and how you use data and labels and have a computer in fair the rules between them for you. You looked at a very simple example where it figured out the relationship between two sets of numbers. Let's now take this to the next level by solving a real problem, computer vision. Computer vision is the field of having a computer understand and label what is present in an image. Consider this slide. When you look at it, you can interpret what a shirt is or what a shoe is, but how would you program for that? If an extra terrestrial who had never seen clothing walked into the room with you, how would you explain the shoes to him? It's really difficult, if not impossible to do right? And it's the same problem with computer vision. So one way to solve that is to use lots of pictures of clothing and tell the computer what that's a picture of and then have the computer figure out the patterns that give you the difference between a shoe, and a shirt, and a handbag, and a coat. That's what you're going to learn how to do in this section. Fortunately, there's a data set called Fashion MNIST which gives a 70 thousand images spread across 10 different items of clothing. These images have been scaled down to 28 by 28 pixels. Now usually, the smaller the better because the computer has less processing to do. But of course, you need to retain enough information to be sure that the features and the object can still be distinguished. If you look at this slide you can still tell the difference between shirts, shoes, and handbags. So this size does seem to be ideal, and it makes it great for training a neural network. The images are also in gray scale, so the amount of information is also reduced. Each pixel can be represented in values from zero to 255 and so it's only one byte per pixel. With 28 by 28 pixels in an image, only 784 bytes are needed to store the entire image. Despite that, we can still see what's in the image and in this case, it's an ankle boot, right?

==Exploring how to use data
Machine Learning depends on having good data to train a system with. In this video you saw a scenario for training a system to recognize fashion images. The data comes from a dataset called Fashion MNIST, and you can learn more about it and explore it in GitHub here. In the next video, you’ll see how to load that data and prepare it for training.
https://github.com/zalandoresearch/fashion-mnist

== Writing code to load training data
So what will handling this look like in code? In the previous lesson, you learned about TensorFlow and Keras, and how to define a super simple neural network with them. In this lesson, you're going to use them to go a little deeper but the overall API should look familiar. The one big difference will be in the data. The last time you had your six pairs of numbers, so you could hard code it. This time you have to load 70,000 images off the disk, so there'll be a bit of code to handle that. Fortunately, it's still quite simple because Fashion-MNIST is available as a data set with an API call in TensorFlow. We simply declare an object of type MNIST loading it from the Keras database. On this object, if we call the load data method, it will return four lists to us. That's the training data, the training labels, the testing data, and the testing labels. Now, what are these you might ask? Well, when building a neural network like this, it's a nice strategy to use some of your data to train the neural network and similar data that the model hasn't yet seen to test how good it is at recognizing the images. So in the Fashion-MNIST data set, 60,000 of the 70,000 images are used to train the network, and then 10,000 images, one that it hasn't previously seen, can be used to test just how good or how bad it is performing. So this code will give you those sets. Then, each set has data, the images themselves and labels and that's what the image is actually of. So for example, the training data will contain images like this one, and a label that describes the image like this. While this image is an ankle boot, the label describing it is the number nine. Now, why do you think that might be? There's two main reasons. First, of course, is that computers do better with numbers than they do with texts. Second, importantly, is that this is something that can help us reduce bias. If we labeled it as an ankle boot, we would be of course biasing towards English speakers. But with it being a numeric label, we can then refer to it in our appropriate language be it English, Chinese, Japanese, or here, even Irish Gaelic.

== The structure of Fashion MNIST data
Here you saw how the data can be loaded into Python data structures that make it easy to train a neural network. You saw how the image is represented as a 28x28 array of greyscales, and how its label is a number. Using a number is a first step in avoiding bias -- instead of labelling it with words in a specific language and excluding people who don’t speak that language! You can learn more about bias and techniques to avoid it here.
https://developers.google.com/machine-learning/fairness-overview/

== Coding a Computer Vision Neural Network
Okay. So now we will look at the code for the neural network definition. Remember last time we had a sequential with just one layer in it. Now we have three layers. The important things to look at are the first and the last layers. The last layer has 10 neurons in it because we have ten classes of clothing in the dataset. They should always match. The first layer is a flatten layer with the input shaping 28 by 28. Now, if you remember our images are 28 by 28, so we're specifying that this is the shape that we should expect the data to be in. Flatten takes this 28 by 28 square and turns it into a simple linear array. The interesting stuff happens in the middle layer, sometimes also called a hidden layer. This is a 128 neurons in it, and I'd like you to think about these as variables in a function. Maybe call them x1, x2 x3, etc. Now, there exists a rule that incorporates all of these that turns the 784 values of an ankle boot into the value nine, and similar for all of the other 70,000. It's too complex a function for you to see by mapping the images yourself, but that's what a neural net does. So, for example, if you then say the function was y equals w1 times x1, plus w2 times x2, plus w3 times x3, all the way up to a w128 times x128. By figuring out the values of w, then y will be nine, when you have the input value of the shoe. You'll see that it's doing something very, very similar to what we did earlier when we figured out y equals 2x minus one. In that case the two, was the weight of x. So, I'm saying y equals w1 times x1, etc. Now, don't worry if this isn't very clear right now. Over time, you will get the hang of it, seeing that it works, and there's also some tools that will allow you to peek inside to see what's going on. The important thing for now is to get the code working, so you can see a classification scenario for yourself. You can also tune the neural network by adding, removing and changing layer size to see the impact. You'll do that in the next exercise. Also, if you want to go further, checkout this tutorial from Andrew on YouTube, which will clarify how dense connected layer's work from the theoretical and mathematical perspective. But for now, let's get back to the code.

== See how it's 
In the next video, Laurence will step you through a workbook where you can see a neural network being trained on Fashion images. After that you’ll be able to try the workbook for yourself!

== Walk through a Notebook for computer vision
Okay. So you just saw how to create a neural network that gives basic computer vision capabilities to recognize different items of clothing. Let's now work through a workbook that has all of the code to do that. You'll then go through this workbook yourself and if you want you can try some exercises. Let's start by importing TensorFlow. I'm going to get the fashion MNIST data using tf.kares.datasets. By calling the load data method, I get training data and labels as well as test data and labels. For more details on these, check back to the previous video. The data for a particular image is a grid of values from zero to 255 with pixel Grayscale values. Using matplotlib, I can plot these as an image to make it easier to inspect. I can also print out the raw values so we can see what they look like. Here you can see the raw values for the pixel numbers from zero to 255, and here you can see the actual image. That was for the first image in the array. Let's take a look at the image at index 42 instead, and we can see the different pixel values and the actual graphic. Our image has values from zero to 255, but neural networks work better with normalized data. So, let's change it to between zero and one simply by dividing every value by 255. In Python, you can actually divide an entire array with one line of code like this. So now we design our model. As explained earlier, there's an input layer in the shape of the data and an output layer in the shape of the classes, and one hidden layer that tries to figure out the roles between them. Now we compile the model to finding the loss function and the optimizer, and the goal of these is as before, to make a guess as to what the relationship is between the input data and the output data, measure how well or how badly it did using the loss function, use the optimizer to generate a new gas and repeat. We can then try to fit the training images to the training labels. We'll just do it for five epochs to be quick. We spend about 25 seconds training it over five epochs and we end up with a loss of about 0.29. That means it's pretty accurate in guessing the relationship between the images and their labels. That's not great, but considering it was done in just 25 seconds with a very basic neural network, it's not bad either. But a better measure of performance can be seen by trying the test data. These are images that the network has not yet seen. You would expect performance to be worse, but if it's much worse, you have a problem. As you can see, it's about 0.345 loss, meaning it's a little bit less accurate on the test set. It's not great either, but we know we're doing something right. Your job now is to go through the workbook, try the exercises and see by tweaking the parameters on the neural network or changing the epochs, if there's a way for you to get it above 0.71 loss accuracy on training data and 0.66 accuracy on test data, give it a try for yourself.

== Get hands-on with computer vision
Now that you’ve seen the workbook, it’s time to try it for yourself.  You can find it here. We’ve also provided a number of exercises you can try at the bottom of the workbook. These will help you poke around and experiment with the code, and will help you with the code you’ll need to write at the end of the week, so it’s really worth spending some time on them! I'd recommend you spend at least 1 hour playing with this workbook. It will be really worth your time!

When you’re done with that, the next thing to do is to explore callbacks, so you can see how to train a neural network until it reaches a threshold you want, and then stop training. You’ll see that in the next video.
https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%204%20-%20Lesson%202%20-%20Notebook.ipynb

== Using Callbacks to control training
A question I often get at this point from programmers in particular when experimenting with different numbers of epochs is, How can I stop training when I reach a point that I want to be at? What do I always have to hard code it to go for certain number of epochs? Well, the good news is that, the training loop does support callbacks. So in every epoch, you can callback to a code function, having checked the metrics. If they're what you want to say, then you can cancel the training at that point. Let's take a look. Okay, so here's our code for training the neural network to recognize the fashion images. In particular, keep an eye on the model.fit function that executes the training loop. You can see that here. What we'll now do is write a callback in Python. Here's the code. It's implemented as a separate class, but that can be in-line with your other code. It doesn't need to be in a separate file. In it, we'll implement the on_epoch_end function, which gets called by the callback whenever the epoch ends. It also sends a logs object which contains lots of great information about the current state of training. For example, the current loss is available in the logs, so we can query it for certain amount. For example, here I'm checking if the loss is less than 0.4 and canceling the training itself. Now that we have our callback, let's return to the rest of the code, and there are two modifications that we need to make. First, we instantiate the class that we just created, we do that with this code. Then, in my model.fit, I used the callbacks parameter and pass it this instance of the class. Let's see this in action.

== See how to implement Callbacks
Experiment with using Callbacks in this notebook -- work through it to see how they perform!
https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%204%20-%20Lesson%204%20-%20Notebook.ipynb

== Walk through a notebook with Callbacks
Let's take a look at the code for callbacks, and see how it works. You can see the code here. Here's the notebook with the code already written, and here's the class that we defined to handle the callback, and here is where we instantiate the callback class itself. Finally, here's where we set up the callback to be called as part of the training loop. As we begin training, note that we asked it to train for five epochs. Now, keep an eye on the losses that trains. We want to break when it goes below 0.4, and by the end of the first epoch we're actually getting close already. As the second epoch begins, it has already dropped below 0.4, but the callback hasn't been hit yet. That's because we set it up for on epoch end. It's good practice to do this, because with some data and some algorithms, the loss may vary up and down during the epoch, because all of the data hasn't yet been processed. So, I like to wait for the end to be sure. Now the epoch has ended, and the loss is 0.3563, and we can see that the training ends, even though we've only done two epochs. Note that we're sure we asked for five epochs and that we ended after two, because the loss is below 0.4, which we checked for in the callback. It's pretty cool right?


## Introduction to Computer vision

### 1.2.1 Fashion MNIST dataset

[Fashion MNIST dataset:](https://github.com/zalandoresearch/fashion-mnist)
- 70k images
- 10 categories
- Images are 28X28 in gray scale *
- Can train a neural net

\* Each pixel can be represented in values from zero to 255 and so it's only one byte per pixel. With 28 by 28 pixels in an image, only 784 bytes are needed to store the entire image. 


### 1.2.2 Load dataset

```python
fashion_mnist = keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels)=fashion_minist.load_data()
```

\* Train/test is 60000/10000.


### 1.2.3 Normilizing
You'll notice that all of the values in the number are between 0 and 255. If we are training a neural network, for various reasons **it's easier if we treat all values as between 0 and 1**, a process called '**normalizing**'...and fortunately in Python it's easy to normalize a list like this without looping. You do it like this:

```python
training_images  = training_images / 255.0
test_images = test_images / 255.0
```
If you tri without normilizaing, the loss will be higher. See more in exercise jupyter notebook below "[Exercise 7](./myExercise/Course_1_Part_4_Lesson_2_Notebook.ipynb)".

### 1.2.4 Code of Neural Network Definition 

```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape(28,28)),
    keras.layers.Dense(128,activation=tf.nn.relu), #middle layer/ hidden layer#
    keras.layers.Dense(10,activation=tf.nn.softmax)
])
```
#### Flatten

**Flatten** takes this 28 by 28 square and turns it into a simple linear array.

Right now our data is 28x28 images, and 28 layers of 28 neurons would be infeasible, so it makes more sense to 'flatten' that 28,28 into a **784x1** (since 28*28=784). 

Instead of wriitng all the code to handle that ourselves, we add the Flatten() layer at the begining, and when the arrays are loaded into the model later, they'll automatically be flattened for us.

#### Middle layer/ Hiddern layer

**Middle layer/ hiddern layer** has 128 neurons. And I'd like you to think about these as variables in a **function**. Maybe call them x1, x2 x3, etc. 

For example, if you then say the **function** was y equals w1 times x1, plus w2 times x2, plus w3 times x3, all the way up to a w128 times x128. 
$$y =w_1 * x_1 + w_2*x_2 + w_3*x_3 + ... + w_{128} * x_{128}$$
By figuring out the values of w, then y will be nine, when you have the input value of the shoe.

![alt text](https://github.com/DayuanTan/AITensorFlowSpecialization/blob/master/img/cv_neural.png)


You can modify the 128 here. For example change it to 1024 neurons. By **adding more Neurons** we have to do more calculations, slowing down the process, but in this case they have a good impact -- we do get more accurate. That doesn't mean it's always a case of 'more is better', you can hit the law of diminishing returns very quickly!


#### Output layer
 The last layer has 10 neurons in it because we have ten classes of clothing in the dataset. They should always match.

**Rule of thumb**-- the **number of neurons** in the last layer should match the **number of classes** you are classifying for. In this case it's the digits 0-9, so there are 10 of them, hence you should have 10 neurons in your final layer.

#### More
More layers, more epochs, usually give us better accuracy. But not always.

Try 15 epochs -- you'll probably get a model with a much better loss than the one with 5.
 
Try 30 epochs -- you might see the loss value stops decreasing, and sometimes increases. This is a **side effect** of something called 'overfitting' which you can learn about [somewhere] and it's something you need to keep an eye out for when training neural networks. There's no point in wasting your time training if you aren't improving your loss, right! :)

```python
model.compile(optimizer = tf.train.AdamOptimizer(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=9)

model.evaluate(test_images, test_labels)
```



*You can also tune the neural network by adding, removing and changing layer size to see the impact. If you want to go further, checkout this tutorial from Andrew on YouTube, which will clarify how dense connected layer's work from the theoretical and mathematical perspective. 
More, plz see Andrew's vedio in Youtube ["What is Neurual Network?"](https://youtu.be/fXOsFF95ifk)*


### 1.2.5 Callback

If you want to stop the training when I reach a desired value?' -- i.e. 95% accuracy might be enough for you. 
You can use callback, so in every epoch, you can callback to a code function, having checked the metrics. If they're what you want to say, then you can cancel the training at that point. See example in exercise jupyter notebook "[Exercise 8](./myExercise/Course_1_Part_4_Lesson_2_Notebook.ipynb)" or [Callback Example](./myExercise/Course_1_Part_4_Lesson_4_Notebook.ipynb).


```python
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss')<0.4):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True
```
In it, we'll implement the **on_epoch_end** function, which gets called by the callback whenever the epoch ends. It also sends a **logs object** which contains lots of great information about the current state of training.

![alt text](https://github.com/DayuanTan/AITensorFlowSpecialization/raw/master/img/callback.png)

### 1.2.6 Try it yourself

[Official code](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%204%20-


