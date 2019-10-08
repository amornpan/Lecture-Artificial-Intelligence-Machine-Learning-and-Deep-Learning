The fashion and this data set was created by [inaudible] and [inaudible]. I think it's really cool that you're already able to implement a neural network to do this fashion classification task. It's just amazing that large data sets like this are readily available to students so that they can learn and it make it really easy to learn. And in this case we saw with just a few lines of code, we were able to build a DNN that allowed you to do this classification of clothing and we got reasonable accuracy with it but it was a little bit of a naive algorithm that we used, right? We're looking at every pixel in every image, but maybe there's ways that we can make it better but maybe looking at features of what makes a shoe a shoe and what makes a handbag a handbag. What do you think? Yeah. So one of the ideas that make these neural networks work much better is to use convolutional neural networks, where instead of looking at every single pixel and say, "Oh, that pixel has value 87, that has value 127." So is this a shoe or is this a hand bag? I don't know. But instead you can look at a picture and say, "Oh, I see shoelaces and a sole." Then, it's probably shoe or say, "I see a handle and rectangular bag beneath that." Probably a handbag. So confidence hopefully, we'll let the students do this. Sure, what's really interesting about convolutions is they sound complicated but they're actually quite straightforward, right? It's a filter that you pass over an image in the same way as if you're doing sharpening, if you've ever done image processing. It can spot features within the image as you've mentioned. With the same paradigm of just data labels, we can let a neural network figure out for itself that it should look for shoe laces and soles or handles in bags and just learn how to detect these things by itself. So shall we see what impact that would have on Fashion MNIST? So in the next video, you'll learn about convolutional neural networks and get to use it to build a much better fashion classifier.

In the previous example, you saw how you could create a neural network called a deep neural network to pattern match a set of images of fashion items to labels. In just a couple of minutes, you're able to train it to classify with pretty high accuracy on the training set, but a little less on the test set. Now, one of the things that you would have seen when you looked at the images is that there's a lot of wasted space in each image. While there are only 784 pixels, it will be interesting to see if there was a way that we could condense the image down to the important features that distinguish what makes it a shoe, or a handbag, or a shirt. That's where convolutions come in. So, what's convolution? You might ask. Well, if you've ever done any kind of image processing, it usually involves having a filter and passing that filter over the image in order to change the underlying image. The process works a little bit like this. For every pixel, take its value, and take a look at the value of its neighbors. If our filter is three by three, then we can take a look at the immediate neighbor, so that you have a corresponding three by three grid. Then to get the new value for the pixel, we simply multiply each neighbor by the corresponding value in the filter. So, for example, in this case, our pixel has the value 192, and its upper left neighbor has the value zero. The upper left value and the filter is negative one, so we multiply zero by negative one. Then we would do the same for the upper neighbor. Its value is 64 and the corresponding filter value was zero, so we'd multiply those out. Repeat this for each neighbor and each corresponding filter value, and would then have the new pixel with the sum of each of the neighbor values multiplied by the corresponding filter value, and that's a convolution. It's really as simple as that. The idea here is that some convolutions will change the image in such a way that certain features in the image get emphasized. So, for example, if you look at this filter, then the vertical lines in the image really pop out. With this filter, the horizontal lines pop out. Now, that's a very basic introduction to what convolutions do, and when combined with something called pooling, they can become really powerful. But simply, pooling is a way of compressing an image. A quick and easy way to do this, is to go over the image of four pixels at a time, i.e, the current pixel and its neighbors underneath and to the right of it. Of these four, pick the biggest value and keep just that. So, for example, you can see it here. My 16 pixels on the left are turned into the four pixels on the right, by looking at them in two-by-two grids and picking the biggest value. This will preserve the features that were highlighted by the convolution, while simultaneously quartering the size of the image. We have the horizontal and vertical axes.

The concepts introduced in this video are available as Conv2D layers and MaxPooling2D layers in TensorFlow. You’ll learn how to implement them in code in the next video…
https://www.tensorflow.org/versions/r1.8/api_docs/python/tf/keras/layers/Conv2D
https://www.tensorflow.org/versions/r1.8/api_docs/python/tf/layers/MaxPooling2D

So now let's take a look at convolutions and pooling in code. We don't have to do all the math for filtering and compressing, we simply define convolutional and pooling layers to do the job for us.
0:12
So here's our code from the earlier example, where we defined out a neural network to have an input layer in the shape of our data, and output layer in the shape of the number of categories we're trying to define, and a hidden layer in the middle. The Flatten takes our square 28 by 28 images and turns them into a one dimensional array.
0:30
To add convolutions to this, you use code like this. You'll see that the last three lines are the same, the Flatten, the Dense hidden layer with 128 neurons, and the Dense output layer with 10 neurons. What's different is what has been added on top of this. Let's take a look at this, line by line.
0:48
Here we're specifying the first convolution. We're asking keras to generate 64 filters for us. These filters are 3 by 3, their activation is relu, which means the negative values will be thrown way, and finally the input shape is as before, the 28 by 28. That extra 1 just means that we are tallying using a single byte for color depth. As we saw before our image is our gray scale, so we just use one byte.
1:13
Now, of course, you might wonder what the 64 filters are. It's a little beyond the scope of this class to define them, but they aren't random. They start with a set of known good filters in a similar way to the pattern fitting that you saw earlier, and the ones that work from that set are learned over time.
1:31
For more details on convolutions and how they work, there's a great set of resources here.

You’ve seen how to add a convolutional 2d layer to the top of your neural network in the previous video. If you want to see more detail on how they worked, check out the playlist at https://bit.ly/2UGa7uH.

Now let’s take a look at adding the pooling, and finishing off the convolutions so you can try them out…
https://bit.ly/2UGa7uH

This next line of code will then create a pooling layer. It's max-pooling because we're going to take the maximum value. We're saying it's a two-by-two pool, so for every four pixels, the biggest one will survive as shown earlier. We then add another convolutional layer, and another max-pooling layer so that the network can learn another set of convolutions on top of the existing one, and then again, pool to reduce the size. So, by the time the image gets to the flatten to go into the dense layers, it's already much smaller. It's being quartered, and then quartered again. So, its content has been greatly simplified, the goal being that the convolutions will filter it to the features that determine the output. A really useful method on the model is the model.summary method. This allows you to inspect the layers of the model, and see the journey of the image through the convolutions, and here is the output. It's a nice table showing us the layers, and some details about them including the output shape. It's important to keep an eye on the output shape column. When you first look at this, it can be a little bit confusing and feel like a bug. After all, isn't the data 28 by 28, so y is the output, 26 by 26. The key to this is remembering that the filter is a three by three filter. Consider what happens when you start scanning through an image starting on the top left. So, for example with this image of the dog on the right, you can see zoomed into the pixels at its top left corner. You can't calculate the filter for the pixel in the top left, because it doesn't have any neighbors above it or to its left. In a similar fashion, the next pixel to the right won't work either because it doesn't have any neighbors above it. So, logically, the first pixel that you can do calculations on is this one, because this one of course has all eight neighbors that a three by three filter needs. This when you think about it, means that you can't use a one pixel margin all around the image, so the output of the convolution will be two pixels smaller on x, and two pixels smaller on y. If your filter is five-by-five for similar reasons, your output will be four smaller on x, and four smaller on y. So, that's y with a three by three filter, our output from the 28 by 28 image, is now 26 by 26, we've removed that one pixel on x and y, and each of the borders. So, next is the first of the max-pooling layers. Now, remember we specified it to be two-by-two, thus turning four pixels into one, and having our x and y. So, now our output gets reduced from 26 by 26, to 13 by 13. The convolutions will then operate on that, and of course, we lose the one pixel margin as before, so we're down to 11 by 11, add another two-by-two max-pooling to have this rounding down, and went down, down to five-by-five images. So, now our dense neural network is the same as before, but it's being fed with five-by-five images instead of 28 by 28 ones. But remember, it's not just one compress five-by-five image instead of the original 28 by 28, there are a number of convolutions per image that we specified, in this case 64. So, there are 64 new images of five-by-five that had been fed in. Flatten that out and you have 25 pixels times 64, which is 1600. So, you can see that the new flattened layer has 1,600 elements in it, as opposed to the 784 that you had previously. This number is impacted by the parameters that you set when defining the convolutional 2D layers. Later when you experiment, you'll see what the impact of setting what other values for the number of convolutions will be, and in particular, you can see what happens when you're feeding less than 784 over all pixels in. Training should be faster, but is there a sweet spot where it's more accurate? Well, let's switch to the workbook, and we can try it out for ourselves.

You’ve now seen how to turn your Deep Neural Network into a Convolutional Neural Network by adding convolutional layers on top, and having the network train against the results of the convolutions instead of the raw pixels. In the next video, you’ll step through a workbook to see how this works…

In the previous video, you looked at convolutions and got a glimpse for how they worked. By passing filters over an image to reduce the amount of information, they then allowed the neural network to effectively extract features that can distinguish one class of image from another. You also saw how pooling compresses the information to make it more manageable. This is a really nice way to improve our image recognition performance. Let's now look at it in action using a notebook. Here's the same neural network that you used before for loading the set of images of clothing and then classifying them. By the end of epoch five, you can see the loss is around 0.29, meaning, your accuracy is pretty good on the training data. It took just a few seconds to train, so that's not bad. With the test data as before and as expected, the losses a little higher and thus, the accuracy is a little lower. So now, you can see the code that adds convolutions and pooling. We're going to do two convolutional layers each with 64 convolution, and each followed by a max pooling layer. You can see that we defined our convolutions to be three-by-three and our pools to be two-by-two. Let's train. The first thing you'll notice is that the training is much slower. For every image, 64 convolutions are being tried, and then the image is compressed and then another 64 convolutions, and then it's compressed again, and then it's passed through the DNN, and that's for 60,000 images that this is happening on each epoch. So it might take a few minutes instead of a few seconds. Now that it's done, you can see that the loss has improved a little. In this case, it's brought our accuracy up a bit for both our test data and with our training data. That's pretty cool, right? Now, let's take a look at the code at the bottom of the notebook. Now, this is a really fun visualization of the journey of an image through the convolutions. First, I'll print out the first 100 test labels. The number nine as we saw earlier is a shoe or boots. I picked out a few instances of this whether the zero, the 23rd and the 28th labels are all nine. So let's take a look at their journey. The Keras API gives us each convolution and each pooling and each dense, etc. as a layer. So with the layers API, I can take a look at each layer's outputs, so I'll create a list of each layer's output. I can then treat each item in the layer as an individual activation model if I want to see the results of just that layer. Now, by looping through the layers, I can display the journey of the image through the first convolution and then the first pooling and then the second convolution and then the second pooling. Note how the size of the image is changing by looking at the axes. If I set the convolution number to one, we can see that it almost immediately detects the laces area as a common feature between the shoes. So, for example, if I change the third image to be one, which looks like a handbag, you'll see that it also has a bright line near the bottom that could look like the soul of the shoes, but by the time it gets through the convolutions, that's lost, and that area for the laces doesn't even show up at all. So this convolution definitely helps me separate issue from a handbag. Again, if I said it's a two, it appears to be trousers, but the feature that detected something that the shoes had in common fails again. Also, if I changed my third image back to that for shoe, but I tried a different convolution number, you'll see that for convolution two, it didn't really find any common features. To see commonality in a different image, try images two, three, and five. These all appear to be trousers. Convolutions two and four seem to detect this vertical feature as something they all have in common. If I again go to the list and find three labels that are the same, in this case six, I can see what they signify. When I run it, I can see that they appear to be shirts. Convolution four doesn't do a whole lot, so let's try five. We can kind of see that the color appears to light up in this case. Let's try convolution one. I don't know about you, but I can play with this all day. Then see what you do when you run it for yourself. When you're done playing, try tweaking the code with these suggestions, editing the convolutions, removing the final convolution, and adding more, etc. Also, in a previous exercise, you added a callback that finished training once the loss had a certain amount. So try to add that here. When you're done, we'll move to the next stage, and that's dealing with images that are larger and more complex than these ones. To see how convolutions can maybe detect features when they aren't always in the same place, like they would be in these tightly controlled 28 by 28 images.

https://colab.sandbox.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%206%20-%20Lesson%202%20-%20Notebook.ipynb
Here’s the notebook that Laurence was using in that screencast. To make it work quicker, go to the ‘Runtime’ menu, and select ‘Change runtime type’. Then select GPU as the hardware accelerator!

Work through it, and try some of the exercises at the bottom! It's really worth spending a bit of time on these because, as before, they'll really help you by seeing the impact of small changes to various parameters in the code. You should spend at least 1 hour on this today!

Once you’re done, go onto the next video and take a look at some code to build a convolution yourself to visualize how it works!

In the previous lessons, you saw the impacts that convolutions and pooling had on your networks efficiency and learning, but a lot of that was theoretical in nature. So I thought it'd be interesting to hack some code together to show how a convolution actually works. We'll also create a little pooling algorithm, so you can visualize its impact. There's a notebook that you can play with too, and I'll step through that here. Here's the notebook for playing with convolutions. It does use a few Python libraries that you may not be familiar with such as cv2. It also has Matplotlib that we used before. If you haven't used them, they're really quite intuitive for this task and they're very very easy to learn. So first, we'll set up our inputs and in particular, import the misc library from SciPy. Now, this is a nice shortcut for us because misc.ascent returns a nice image that we can play with, and we don't have to worry about managing our own. Matplotlib contains the code for drawing an image and it will render it right in the browser with Colab. Here, we can see the ascent image from SciPy. Next up, we'll take a copy of the image, and we'll add it with our homemade convolutions, and we'll create variables to keep track of the x and y dimensions of the image. So we can see here that it's a 512 by 512 image. So now, let's create a convolution as a three by three array. We'll load it with values that are pretty good for detecting sharp edges first. Here's where we'll create the convolution. We iterate over the image, leaving a one pixel margin. You'll see that the loop starts at one and not zero, and it ends at size x minus one and size y minus one. In the loop, it will then calculate the convolution value by looking at the pixel and its neighbors, and then by multiplying them out by the values determined by the filter, before finally summing it all up. Let's run it. It takes just a few seconds, so when it's done, let's draw the results. We can see that only certain features made it through the filter. I've provided a couple more filters, so let's try them. This first one is really great at spotting vertical lines. So when I run it, and plot the results, we can see that the vertical lines in the image made it through. It's really cool because they're not just straight up and down, they are vertical in perspective within the perspective of the image itself. Similarly, this filter works well for horizontal lines. So when I run it, and then plot the results, we can see that a lot of the horizontal lines made it through. Now, let's take a look at pooling, and in this case, Max pooling, which takes pixels in chunks of four and only passes through the biggest value. I run the code and then render the output. We can see that the features of the image are maintained, but look closely at the axes, and we can see that the size has been halved from the 500's to the 250's. For fun, we can try the other filter, run it, and then compare the convolution with its pooled version. Again, we can see that the features have not just been maintained, they may have also been emphasized a bit. So that's how convolutions work. Under the hood, TensorFlow is trying different filters on your image and learning which ones work when looking at the training data. As a result, when it works, you'll have greatly reduced information passing through the network, but because it isolates and identifies features, you can also get increased accuracy. Have a play with the filters in this workbook and see if you can come up with some interesting effects of your own.


To try this notebook for yourself, and play with some convolutions, here’s the notebook. Let us know if you come up with any interesting filters of your own!
https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%206%20-%20Lesson%203%20-%20Notebook.ipynb

As before, spend a little time playing with this notebook. Try different filters, and research different filter types. There's some fun information about them here: https://lodev.org/cgtutor/filtering.html
https://lodev.org/cgtutor/filtering.html

## Enhancing Vision with Convolutional Neural Networks

### 1.3.1 Convolution and pooling

#### 1.3.1.1 Convolution

![alt_text](https://github.com/DayuanTan/AITensorFlowSpecialization/raw/master/img/convolution.png)

#### Explain how convolution works:

For every pixel, take its value, and take a look at the value of its neighbors. If our **filter** is 3x3, then we can take a look at the immediate neighbor, so that you have a corresponding 3x3 grid. Then to get the new value for the pixel, we simply multiply each neighbor by the corresponding value in the filter. 

So, for example, in this case, our pixel has the value 192, and its upper left neighbor has the value zero. The upper left value and the filter is negative one, so we multiply zero by negative one. Then we would do the same for the upper neighbor. Its value is 64 and the corresponding filter value was zero, so we'd multiply those out. Repeat this for each neighbor and each corresponding filter value, and would then have the new pixel with the sum of each of the neighbor values multiplied by the corresponding filter value, and that's a **convolution**. **A technique to isolate features in images**.

#### Convolution is used for emphasizing

The idea here is that some convolutions will change the image in such a way that certain features in the image get emphasized. 

So, for example, if you look at this filter, then the vertical lines in the image really pop out: 
![alt_text](https://github.com/DayuanTan/AITensorFlowSpecialization/raw/master/img/cnnverticalEmphasize.png)

With this filter, the horizontal lines pop out:
![alt_text](https://github.com/DayuanTan/AITensorFlowSpecialization/raw/master/img/cnnhorizontalEmphasize.png)


------

#### Convolution combined with pooling, they can become really powerful. Usually adding convolution layers and pooling layers will make model more accurate, empirically.

------

#### 1.3.1.2 Pooling

![alt_text](https://github.com/DayuanTan/AITensorFlowSpecialization/raw/master/img/cnnpooling.png)

Simply, **pooling is a way of compressing an image**. **A technique to reduce the information in an image while maintaining features**.

Of these four, pick the biggest value and keep just that.

#### Purpose of pooling

This will preserve the features that were highlighted by the convolution, while simultaneously quartering the size of the image. We have the horizontal and vertical axes.

------

### 1.3.2 Convolution & pooling in code

For **convolution** and **pooling** in code, we just need to add few layers before flatterning layer.

![alt_text](https://github.com/DayuanTan/AITensorFlowSpecialization/raw/master/img/cnncode.png)

- **Conv2D** -- First **convolution layer**, asking keras to generate **64 filters** for us. Those filters are not random. They start with a set of **known good filters** in a similar way **to the pattern** (fitting that you saw earlier). The ones (that work) from that set (are learned over time). These **filters** are 3 by 3, their **activation** is **relu**, which means the negative values will be thrown way, and finally the **input shape** is as before, the **28 by 28**. That extra **1** just means that we are tallying(计数 理货) using a single byte for **color depth**. As we saw before our image is our gray scale, so we just use one byte.

- **MaxPooling2D** -- First **pooling layer**. **Max-pooling**: take the maximum value. It's a **two-by-two pool**, so for every four pixels, the **biggest** one will survive as shown earlier. 

- **Conv2D** -- Second **convolution layer**.

- **MaxPooling2D** -- Second **pooling layer**.

So, by the time the image gets to the **flatten** to go into the **dense** layers, it's already much **smaller**. It's being quartered, and then quartered again. So, its content has been greatly simplified.

The **last 3 layers** are same as before in 1.2. 


------

### 1.3.3  model.summary()

Allows you to inspect the layers of the model, and see the journey of the image through the convolutions, and here is the output.

![alt_text](https://github.com/DayuanTan/AITensorFlowSpecialization/raw/master/img/cnnmodelsummary.png)

- **First** **line**, the output shape **isn't** the data 28 by 28, so *y* is the output, **26 by 26**. Because logically, the first pixel that you can do calculations on is this one, because this one of course has all eight neighbors that a three by three filter needs. 

So the output of the convolution will be two pixels smaller on x, and two pixels smaller on y. 

If your filter is **five-by-five** for similar reasons, your output will be **four** smaller on x, and **four** smaller on y. So, that's y with a **three by three** filter, our output from the 28 by 28 image, is now 26 by 26, we've removed that one pixel on x and y, and each of the borders.

![alt_text](https://github.com/DayuanTan/AITensorFlowSpecialization/raw/master/img/cnnfirstpixel.png)

- **Second** **line** (the first pooling layer), remember we specified it to be **two-by-two**, thus turning four pixels into one, and having our x and y. So, now our output gets reduced from **26 by 26**, to **13 by 13**.

- **Forth** **line**, input is **11 by 11**, add another two-by-two max-pooling to have this rounding down, and went down, down to **five-by-five** images **(pay attention is not 6x6)**.


- **Fifth line**, the input is **not** just **one** compress five-by-five image instead of the original 28 by 28, there are **a number of convolutions per image** that we specified, in this case **64**. So, there are 64 new images of five-by-five that had been fed in. 

Flatten that out and you have (5x5=)25 pixels times 64, which is **1600** , as opposed to the **784**(=28x28) that you had previously.

### 1.3.4 Try it yourself (Fashion MNIST)

[Offical code](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%206%20-%20Lesson%202%20-%20Notebook.ipynb)

[my code](./myExercise/Course_1_Part_6_Lesson_2_Notebook.ipynb)

**Fashion MNIST Exercise 1**: Try change 32 to 64 filters in first convolution layer. It takes longer time but the accuracy is better.
```python
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
```

**Fashion MNIST Exercise 3**: add more convolution layers (3 layers). It takes longer time but the accuracy is worse.
```python
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
```

### 1.3.5 More exercise about convolution and pooling

[Official code](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%206%20-%20Lesson%203%20-%20Notebook.ipynb)

[More introduction and explanation about image processing:](https://lodev.org/cgtutor/filtering.html)

- Introduction
- Convolution
- Blur
- Gaussian Blur
- Motion Blur
- Find Edges
- Sharpen
- Emboss
- Mean and Median Filter
- Conclusion

[My code](./myExercise/Convolutions_Sidebar.ipynb)

------

### 1.3.6 Exercise 3 - Handwriting recognition with CNN (MNIST)

[Offical code](./myExercise/official_Exercise_3_Answer.ipynb)

[My code](./myExercise/Exercise_3_Question.ipynb)
We can see after adding convolution and pooling layers, the accuracy is much better than 1.2 Exercise. 

**Output** of 1.2 Simple neural network **V.S.** 1.3 Add convolution and pooling:

```
Epoch 1/20
60000/60000 [==============================] - 27s 444us/sample - loss: 0.1793 - acc: 0.9447
Epoch 2/20
60000/60000 [==============================] - 27s 443us/sample - loss: 0.0798 - acc: 0.9761
Epoch 3/20
60000/60000 [==============================] - 27s 445us/sample - loss: 0.0570 - acc: 0.9814
Epoch 4/20
60000/60000 [==============================] - 27s 443us/sample - loss: 0.0421 - acc: 0.9866
Epoch 5/20
60000/60000 [==============================] - 27s 456us/sample - loss: 0.0346 - acc: 0.9890
Epoch 6/20
59968/60000 [============================>.] - ETA: 0s - loss: 0.0293 - acc: 0.9908
Reached 99% accuracy so cancelling training!
60000/60000 [==============================] - 28s 463us/sample - loss: 0.0295 - acc: 0.9908
10000/10000 [==============================] - 1s 118us/sample - loss: 0.0937 - acc: 0.9766
Out[0]:
[0.09371699871601905, 0.9766]
```

```
_________________________________________________________________
Epoch 1/20
60000/60000 [==============================] - 141s 2ms/sample - loss: 0.1091 - acc: 0.9659
Epoch 2/20
60000/60000 [==============================] - 141s 2ms/sample - loss: 0.0421 - acc: 0.9870
Epoch 3/20
60000/60000 [==============================] - 140s 2ms/sample - loss: 0.0282 - acc: 0.9918
Epoch 4/20
60000/60000 [==============================] - 140s 2ms/sample - loss: 0.0236 - acc: 0.9929
Epoch 5/20
60000/60000 [==============================] - 141s 2ms/sample - loss: 0.0177 - acc: 0.9951
Epoch 6/20
60000/60000 [==============================] - 140s 2ms/sample - loss: 0.0142 - acc: 0.9958
Epoch 7/20
60000/60000 [==============================] - 140s 2ms/sample - loss: 0.0134 - acc: 0.9956
Epoch 8/20
60000/60000 [==============================] - 141s 2ms/sample - loss: 0.0116 - acc: 0.9965
Epoch 9/20
60000/60000 [==============================] - 140s 2ms/sample - loss: 0.0109 - acc: 0.9968
Epoch 10/20
59968/60000 [============================>.] - ETA: 0s - loss: 0.0070 - acc: 0.9980
Reached 99.8% accuracy so cancelling training!
60000/60000 [==============================] - 141s 2ms/sample - loss: 0.0070 - acc: 0.9980
10000/10000 [==============================] - 6s 607us/sample - loss: 0.0430 - acc: 0.9898
[0.04297752619532357, 0.9898]
```




