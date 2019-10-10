So, there we saw that the classifier that we built for classifying fashion, like using convolutions were able to make it more efficient and to make it more accurate. I thought that was really, really cool, but it's still very limited in the scenario because all of our images are 28 by 28 and the subject is actually centered. And while it's a fabulous dataset for learning, it's like when we start getting into real-world images and complex images that maybe we need to go a little bit further. What do you think? I think it's really cool that taking the core idea of a confinet allows you to implement an algorithm to confine not just handbags right in the middle of the image but anywhere in the image, so it could be carried by someone on the left or the right of a much bigger and, say, a one-megapixel image. This is 1000 by 1000 pixels. Also for many applications rather than using grayscale, want to use color images- All right. And the same core ideas but with a bigger dataset, bigger images in similar labels lets you do that. All right. So, the technique that you're learning in this, is really really helping you to be able to succeed in these more real-world scenarios. So, I know you've been working on a dataset on horses- Yeah. And humans. Yeah, that's been a lot of fun. I've been working on a dataset that's a number of images of horses and they're moving around the image and they're in different poses, and humans in the same way and diverse humans male, female, different races, that kind of thing to see if we can build a binary classifier between the two of them. But what was really interesting about this is that they're all computer-generated images, but we can use them to classify real photos. I had a lot of fun with that. So, I think there'll be a fun exercise for you to work on as well. And if you're ever wondering of these algorithms you're learning whether this is the real stuff, the algorithims you're learning is really the real stuff that is used today in many commercial applications. For example, if you look at the way a real self-driving car today uses cameras to detect other vehicles or pedestrians to try to avoid them, they use convolutional neural networks for that part of the task, very similar to what you are learning. And in fact, in other contexts, I've heard you speak about using a convolutional neural network. To take a picture, for example. Yeah, we can take a picture of a crop and try to tell if it has a disease coming. So, that was really cool. Oh, thank you, thank you. That's really fun. So, in the next video, you'll learn how to apply convolutional neural networks to these much bigger and more complex images. Please go on to the next video.


As Andrew and Laurence discussed, the techniques you’ve learned already can apply to complex images, and you can start solving real scenarios with them. They discussed how it could be used, for example, in disease detection with the Cassava plant, and you can see a video demonstrating that here. Once you’ve watched that, move onto the next lesson!
https://www.youtube.com/watch?v=NlpS-DhayQA

To this point, you built an image classifier that worked using a deep neural network and you saw how to improve its performance by adding convolutions. One limitation though was that it used a dataset of very uniform images. Images of clothing that was staged and framed in 28 by 28. But what happens when you use larger images and where the feature might be in different locations? For example, how about these images of horses and humans? They have different sizes and different aspect ratios. The subject can be in different locations. In some cases, there may even be multiple subjects. In addition to that, the earlier examples with a fashion data used a built-in dataset. All of the data was handily split into training and test sets for you and labels were available. In many scenarios, that's not going to be the case and you'll have to do it for yourself. So in this lesson, we'll take a look at some of the APIs that are available to make that easier for you. In particular, the image generator in TensorFlow. One feature of the image generator is that you can point it at a directory and then the sub-directories of that will automatically generate labels for you. So for example, consider this directory structure. You have an images directory and in that, you have sub-directories for training and validation. When you put sub-directories in these for horses and humans and store the requisite images in there, the image generator can create a feeder for those images and auto label them for you. So for example, if I point an image generator at the training directory, the labels will be horses and humans and all of the images in each directory will be loaded and labeled accordingly. Similarly, if I point one at the validation directory, the same thing will happen. So let's take a look at this in code. The image generator class is available in Keras.preprocessing.image. You can then instantiate an image generator like this. I'm going to pass rescale to it to normalize the data. You can then call the flow from directory method on it to get it to load images from that directory and its sub-directories. It's a common mistake that people point the generator at the sub-directory. It will fail in that circumstance. You should always point it at the directory that contains sub-directories that contain your images. The names of the sub-directories will be the labels for your images that are contained within them. So make sure that the directory you're pointing to is the correct one. You put it in the second parameter like this. Now, images might come in all shapes and sizes and unfortunately for training a neural network, the input data all has to be the same size, so the images will need to be resized to make them consistent. The nice thing about this code is that the images are resized for you as they're loaded. So you don't need to preprocess thousands of images on your file system. But you could have done that if you wanted to. The advantage of doing it at runtime like this is that you can then experiment with different sizes without impacting your source data. While the horses and humans dataset is already in 300 by 300, when you use other datasets they may not always be uniformly sized. So this is really useful for you. The images will be loaded for training and validation in batches where it's more efficient than doing it one by one. Now, there's a whole science to calculating batch size that's beyond the scope of this course, but you can experiment with different sizes to see the impact on the performance by changing this parameter. Finally, there's the class mode. Now, this is a binary classifier i.e. it picks between two different things; horses and humans, so we specify that here. Other options in particular for more than two things will be explored later in the course. The validation generator should be exactly the same except of course it points at a different directory, the one containing the sub-directories containing the test images. When you go through the workbook shortly, you'll see how to download the images as a zip, and then sort them into training and test sub-directories, and then put horses and humans sub-directories in each. That's just pure Python. It's not TensorFlow or any other deep learning stuff. But it's all explained for you in the notebook.

Now that you’ve seen how an ImageGenerator can flow images from a directory and perform operations such as resizing them on the fly, the next thing to do is design the neural network to handle these more complex images. You’ll see that in the next video.

So let's now take a look at the definition of the neural network that we'll use to classify horses versus humans. It's very similar to what you just used for the fashion items, but there are a few minor differences based on this data, and the fact that we're using generators. So here's the code. As you can see, it's the sequential as before with convolutions and pooling before we get to the dense layers at the bottom. But let's highlight some of the differences. First of all, you'll notice that there are three sets of convolution pooling layers at the top. This reflects the higher complexity and size of the images. Remember our earlier our 28 by 28.5 to 13 and then five before flattening, well, now we have 300 by 300. So we start at 298 by 298 and then have that etc., etc. until by the end, we're at a 35 by 35 image. We can even add another couple of layers to this if we wanted to get to the same ballpark size as previously, but we'll keep it at three for now. Another thing to pay attention to is the input shape. We resize their images to be 300 by 300 as they were loaded, but they're also color images. So there are three bytes per pixel. One byte for the red, one for green, and one for the blue channel, and that's a common 24-bit color pattern. If you're paying really close attention, you can see that the output layer has also changed. Remember before when you created the output layer, you had one neuron per class, but now there's only one neuron for two classes. That's because we're using a different activation function where sigmoid is great for binary classification, where one class will tend towards zero and the other class tending towards one. You could use two neurons here if you want, and the same softmax function as before, but for binary this is a bit more efficient. If you want you can experiment with the workbook and give it a try yourself. Now, if we take a look at our model summary, we can see the journey of the image data through the convolutions The 300 by 300 becomes 298 by 298 after the three by three filter, it gets pulled to 149 by 149 which in turn gets reduced to 73 by 73 after the filter that then gets pulled to 35 by 35, this will then get flattened, so 64 convolutions that are 35 squared and shape will get fed into the DNN. If you multiply 35 by 35 by 64, you get 78,400, and that's the shape of the data once it comes out of the convolutions. If we had just fed raw 300 by 300 images without the convolutions, that would be over 900,000 values. So we've already reduced it quite a bit.

Now that you’ve designed the neural network to classify Horses or Humans, the next step is to train it from data that’s on the file system, which can be read by generators. To do this, you don’t use model.fit as earlier, but a new method call: model.fit_generator. In the next video you’ll see the details of this.

Okay, we'll now compile the model and, as always, we have a loss function and an optimizer. When classifying the ten items of fashion, you might remember that your loss function was a categorical cross entropy. But because we're doing a binary choice here, let's pick a binary_crossentropy instead. Also, earlier we used an Adam optimizer. Now, you could do that again, but I thought it would be fun to use the RMSprop, where you can adjust the learning rate to experiment with performance. To understand learning rate and how all that fits together, check out this great video from deeplearning.ai that goes into it in a lot more detail.
0:36
For now, I'm not going to go into the details in this course. Okay, next up is the training, now, this looks a little different than before when you called model.fit. Because now you call model.fit_generator, and that's because we're using a generator instead of datasets. Remember the image generator from earlier, let's look at each parameter in detail. The first parameter is the training generator that you set up earlier. This streams the images from the training directory. Remember the batch size you used when you created it, it was 20, that's important in the next step. There are 1,024 images in the training directory, so we're loading them in 128 at a time. So in order to load them all, we need to do 8 batches. So we set the steps_per_epoch to cover that.
1:23
Here we just set the number of epochs to train for. This is a bit more complex, so let's use, say, 15 epochs in this case. And now we specify the validation set that comes from the validation_generator that we also created earlier. It had 256 images, and we wanted to handle them in batches of 32, so we will do 8 steps.
1:45
And the verbose parameter specifies how much to display while training is going on. With verbose set to 2, we'll get a little less animation hiding the epoch progress. Once the model is trained, you will, of course, want to do some prediction on the model. And here's the code to do that, let's look at it piece by piece.
2:04
So these parts are specific to Colab, they are what gives you the button that you can press to pick one or more images to upload. The image paths then get loaded into this list called uploaded. The loop then iterates through all of the images in that collection. And you can load an image and prepare it to input into the model with this code. Take note to ensure that the dimensions match the input dimensions that you specified when designing the model. You can then call model.predict, passing it the details, and it will return an array of classes. In the case of binary classification, this will only contain one item with a value close to 0 for one class and close to 1 for the other.
2:43
Later in this course you'll see multi-class classification with Softmax. Where you'll get a list of values with one value for the probability of each class and all of the probabilities adding up to 1.

Now you’ve gone through the code to define the neural network, train it with on-disk images, and then predict values for new images. Let’s see this in action in a workbook. In the next video, Laurence will step you through the workbook, and afterwards, you can try it for yourself!

Okay. So you've just seen how to get started with creating a neural network in Keras that uses the image generator to automatically load and label your files based on their subdirectories. Now, let's see how we can use that to build a horses or humans classifier with a convolutional neural network. This is the first notebook you can try. To start, you'll download the zip file containing the horses and humans data. Once that's done, you can unzip it to the temp directory on this virtual machine. The zip file contain two folders; one called filtered horses, and one called filtered humans. When it was unzipped, these were created for you. So we'll just point a couple of variables at them, and then we can explore the files by printing out some of the filenames. Now, these could be used to generate labels, but we won't need that if we use the Keras generator. If you wanted to use this data without one, a filenames will have the labels in them of course though. We'll print out the number of images that we have to work with, and there's a little over 1000 of them, and now we can display a few random images from the dataset. Here, we can see eight horses and eight humans. An interesting aspect of this dataset is that all of the images are computer-generated. I've rendered them to be as photo-real as possible, but there'll be actually used to classify real pictures of horses and people, and here's a few more images just to show some of the diversity. Let's start building the model. First, we'll import TensorFlow, and now we'll build the layers. We have quite a few convolutions here because our source images are quite large, are 300 by 300. Later we can explore the impact of reducing their size and needing less convolutions. We can print the summary of the layers, and here we can see by the time we reach the dense network, the convolutions are down to seven-by-seven. Okay. Next up, we'll compiler network. It's using binary cross entropy as the loss, binary because we're using just two classes, and the optimizer is an RMSprop that allows us to tweak the learning rate. Don't worry if you don't fully understand these yet, there are links out to content about them where you can learn more.

Now that you’ve learned how to download and process the horses and humans dataset, you’re ready to train. When you defined the model, you saw that you were using a new loss function called ‘Binary Crossentropy’, and a new optimizer called RMSProp. If you want to learn more about the type of binary classification we are doing here, check out this great video from Andrew!
https://gombru.github.io/2018/05/23/cross_entropy_loss/
https://www.tensorflow.org/api_docs/python/tf/train/RMSPropOptimizer
http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
https://www.youtube.com/watch?v=eqEc66RFY0I&t=6s

Next up is where we use the ImageDataGenerator. We instantiate it and we scale our images to 1 over 255, which then normalizes their values. We then point it at the main directory where we see the unzipped files.
0:16
We can see that it finds all of the images, and has assigned them to two classes, because they were two sub directories. We'll now train the neural network for 15 epochs, it will take about two minutes.
0:29
Each epoch is loading the data, calculating the convolutions and then trying to match the convolutions to labels. As you can see, the accuracy mostly increases but it will occasionally deep, showing the gradient ascent of the learning actually in action.
0:44
It's always a good idea to keep an eye on fluctuations in this figure. And if there are too wild, you can adjust the learning rate.
0:52
Remember the parameter to RMS prop when you compile the model, that's where you'd tweak it. It's also going pretty fast, because right here, I'm training on a GPU machine.
1:02
By the time we get to epoch 15, we can see that our accuracy is about 0.9981, which is really good. But remember, that's only based on the data that the network has already seen during training, which is only about 1,000 images. So don't get lulled into a false sense of security.
1:21
Let's have a bit of fun with the model now and see if we can predict the class for new images that it hasn't previously seen.
1:28
Let's go to Pixabay, and see what we can find. I'll search for horses, and there's lots of horses, so let's pick this one. It's a white horse running in the snow. I'm going to download it to my file system. I'm now going to go back to the notebook, and I'm going to upload the image from my file system. And we'll see that it gets uploaded, and it's classified as a horse. So let's try another one. Like this one here. Which I'll then upload to the notebook, and we'll see that it's also classified as a horse.
1:58
I'll now go back to Pixabay and search for person, and pick this image of a girl sitting on a bench. I'll download it to my file system, upload it to the neural network, and we can see that this is also correctly classified as a human.
2:15
Let's do one more. I'll go back to the list of results on Pixabay, and pick this image of a girl. As before, I'll download it to my file system and I'll upload it to the neural network and we'll also see that it's still detects a human in the image.
2:31
Now one other thing that I can do with this script is upload multiple files and have it classify all of them at once. And here we can see all of the classifications. We have four out four correct.
2:42
This notebook also includes some visualizations of the image as it passes through the convolutions. You can give it a try with this script.
2:50
Here you can see where a human image was convolved and features such as the legs really lit up. And if I run it again, we can see another human with similar features. Also the hair is very distinctive. Have a play with it for yourself and see what you discover.
3:07
So there, we saw a convolutional neural network create a classifier to horses or humans using a set of about 1,000 images. The four images we tested all worked, but that's not really scalable. And the next video, we'll see how we can add a validation set to the training and have it automatically measure the accuracy of the validation set, too.

Now it’s your turn. You can find the notebook here. Work through it and get a feel for how the ImageGenerator is pulling the images from the file system and feeding them into the neural network for training. Have some fun with the visualization code at the bottom!

In earlier notebooks you tweaked parameters like epochs, or the number of hidden layers and neurons in them. Give that a try for yourself, and see what the impact is. Spend some time on this.

Once you’re done, move to the next video, where you can validate your training against a lot of images!
https://colab.sandbox.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%208%20-%20Lesson%202%20-%20Notebook.ipynb

In the previous video, you saw how to build a convolutional neural network that classified horses against humans. When you are done, you then did a few tests using images that you downloaded from the web. In this video, you'll see how you can build validation into the training loop by specifying a set of validation images, and then have TensorFlow do the heavy lifting of measuring its effectiveness with that same. As before we download the dataset, but now will also download the separate validation dataset. We'll unzip into two separate folders, one for training, one for validation. We'll create some variables that pointed our training and validation subdirectories, and we can check out the filenames. Remember that the filenames may not always be reliable for labels. For example, here the validation horse labels aren't named as such while the human ones are. We can also do a quick check on whether we got all the data, and it looks good so we think we can proceed. We can display some of the training images as we did before, and let's just go straight to our model. Here we can import TensorFlow, and here we define the layers in our model. It's exactly the same as last time. We'll then print the summary of our model, and you can see that it hasn't changed either. Then we'll compile the model with the same parameters as before. Now, here's where we can make some changes. As well as an image generator for the training data, we now create a second one for the validation data. It's pretty much the same flow. We create a validation generator as an instance of image generator, re-scale it to normalize, and then pointed at the validation directory. When we run it, we see that it picks up the images and the classes from that directory. So now let's train the network. Note the extra parameters to let it know about the validation data. Now, at the end of every epoch as well as reporting the loss and accuracy on the training, it also checks the validation set to give us loss in accuracy there. As the epochs progress, you should see them steadily increasing with the validation accuracy being slightly less than the training. It should just take about another two minutes. Okay. Now that we've reached epoch 15, we can see that our accuracy is about 97 percent on the training data, and about 85 percent on the validation set, and this is as expected. The validation set is data that the neural network hasn't previously seen, so you would expect it to perform a little worse on it. But let's try some more images starting with this white horse.
2:42
We can see that it was misclassified as a human. Okay, let's try this really cute one.
2:54
We can see that's correctly classified as a horse. Okay, let's try some people.
3:04
Let's try this woman in a blue dress. This is really interesting picture because she has her back turned, and her legs are obscured by the dress, but she's correctly classified as a human. Okay, here's a tricky one. To our eyes she's human, but will the wings confuse the neural network?
3:28
And they do, she is mistaken for a horse. It's understandable though particularly as the training set has a lot of white horses against the grassy background.
3:40
How about this one? It has both a horse and the human in it,
3:48
but it gets classified as a horse. We can see the dominant features in the image are the horse, so it's not really surprising. Also there are many white horses in the training set, so it might be matching on them. Okay one last one. I couldn't resist this image as it's so adorable,
4:11
and thankfully it's classified as a horse. So, now we saw the training with a validation set, and we could get a good estimate for the accuracy of the classifier by looking at the results with a validation set. Using these results and understanding where and why some inferences fail, can help you understand how to modify your training data to prevent errors like that. But let's switch gears in the next video. We'll take a look at the impact of compacting your data to make training quicker.

Now you can give it a try for yourself. Here’s the notebook the Laurence went through in the video. Have a play with it to see how it trains, and test some images yourself! Once you’re done, move onto the next video where you’ll compact your data to see the impact on training.
https://colab.sandbox.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%208%20-%20Lesson%203%20-%20Notebook.ipynb


The images in the horses are humans dataset are all 300 by 300 pixels. So we had quite a few convolutional layers to reduce the images down to condensed features. Now, this of course can slow down the training. So let's take a look at what would happen if we change it to a 150 by a 150 for the images to have a quarter of the overall data and to see what the impact would be. We'll start as before by downloading and unzipping the training and test sets.
0:29
Then we'll point some variables in the training and test sets before setting up the model. First, we'll import TensorFlow and now we'll define the layers for the model. Note that we've changed the input shape to be 150 by 150, and we've removed the fourth and fifth convolutional max pool combinations. Our model summary now shows the layer starting with the 148 by 148, that was the result of convolving the 150 by 150. We'll see that at the end, we end up with a 17 by 17 by the time we're through all of the convolutions and pooling. We'll compile our model as before, and we'll create our generators as before, but note that the target size has now changed to 150 by 150.
1:12
Now we can begin the training, and we can see that after the first epoch that the training is fast, and accuracy and validation aren't too bad either. The training continues and both accuracy values will tick up.
1:33
Often, you'll see accuracy values that are really high like 1.000, which is likely a sign that you're overfitting. We reach the end, I have really high accuracy on the test data, about 0.99, which is much too high. The validation set is about 0.84, which is pretty good, but let's put it to the test with some real images.
1:56
Let's start with this image of the girl and the horse. It still classifies as a horse. Next, let's take a look at this cool horsey,
2:09
and who's still correctly categorized.
2:17
These cuties are also correctly categorized,
2:23
but this one is still wrongly categorized. But the most interesting I think is this woman. When we use 300 by 300 before and more convolutions, she was correctly classified. But now, she isn't. This is a great example of the importance of measuring your training data against a large validation set, inspecting where it got it wrong and seeing what you can do to fix it. Using this smaller set is much cheaper to train, but then errors like this woman with her back turned and her legs obscured by the dress will happen, because we don't have that data in the training set. That's a nice hint about how to edit your dataset for the best effect in training.

Try this version of the notebook where Laurence compacted the images. You can see that training times will improve, but that some classifications might be wrong! Experiment with different sizes -- you don’t have to use 150x150 for example!
https://colab.sandbox.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%208%20-%20Lesson%204%20-%20Notebook.ipynb


## Using Real-world Images

### 1.4.1 Understanding ImageGenerator

What if we use larger images and where the feature might be in different locations?
Like this:
![alt_text](https://github.com/DayuanTan/AITensorFlowSpecialization/raw/master/img/4horsehuman.png)
**Difference**: They have different sizes and different aspect ratios纵横比. The subject can be in different locations. In some cases, there may even be multiple subjects. 

#### 1.4.1.1 ImageGenerator

**Feature**:

One **feature** of the **image generator** is that you can point it at a directory and then the sub-directories of that will automatically generate labels for you. 

You can use the image generator to automatically load and label your files based on their subdirectories. Images 'i.jpg', '2.jpg', '3.jpg' will be labelled with "Horses".

![alt_text](https://github.com/DayuanTan/AITensorFlowSpecialization/raw/master/img/4imagegenerator.png)

Be careful the **directory** must be correct. In this figure, the **train_dir** is "Training" and the **validation_dir** is "Validation". The **names of the sub-directories** will be the **labels** for your images that are contained within them. So the 'Horses' sub-directory should contain all horses images.

```python
from tensorflow.keras.preprocessing.image
import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255) #pass rescale to it to normalize the data
```
```python
#call the flow from directory method on it to get it to load images from that directory and its sub-directories. 
train_generator = train_datagen.flow_from_directory(
    train_dir, 
```
You should always point it at the **directory** that contains sub-directories that contain your images. The **names of the sub-directories** will be the **labels** for your images that are contained within them.
```python
    target_size=(300, 300), 
```
Images might come in all shapes and sizes and unfortunately for training a neural network, the **input data all has to be the same size**, so the images will need to be **resized** to make them consistent. The nice thing about this code is that the images are resized for you as they're loaded. The advantage of doing it at runtime like this is that you can then experiment with different sizes **without impacting your source data**. 
```python
    batch_size=128,
```
The images will be loaded for training and validation in **batches** where it's more **efficient** than doing it one by one. You can try different batch size.
```python
    class_mode='binary'
```
This is a **binary** **classifier** i.e. it picks between two different things; horses and humans, so we specify that here. There are also other options. 

```python
)


test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(300, 300),
    batch_size=32,
    class_mode='binary'
)

```

![alt_text](https://github.com/DayuanTan/AITensorFlowSpecialization/raw/master/img/4train.png)
![alt_text](https://github.com/DayuanTan/AITensorFlowSpecialization/raw/master/img/4validation.png)

------

### 1.4.2 Defining a ConvNet to use complex images

```python
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

We have **3 sets** of convolution pooling layers. This reflects the higher complexity and size of the images.

#### Color images
We resize their images to be 300 by 300 as they were loaded, but they're also **color images**. So there are **three bytes per pixel**. One byte for the **red**, one for **green**, and one for the **blue** channel, and that's a common 24-bit color pattern.

#### Activation function (Sigmoid or Softmax)

```python
#old:
tf.keras.layers.Dense(2, activation='softmax') # 2 classifications, human or horse
#new:
tf.keras.layers.Dense(1, activation='sigmoid')
```

Remember before when you created the output layer, you had one neuron per class, but now there's only one neuron for two classes. That's because we're using a different **activation function** where **sigmoid** is great for **binary classification**, where one class will tend towards zero and the other class tending towards one. You could use two neurons here if you want, and the same softmax function as before, but for binary this is a bit more efficient.

- **Sigmoid**

![alt_text](https://github.com/DayuanTan/AITensorFlowSpecialization/raw/master/img/4sigmoid.png)

- **Softmax**

    **Multi-class classification** with **Softmax**. Where you'll get **a list of values** with one value for the **probability of each class** and **all of the probabilities adding up to 1**.

------

### 1.4.3 Model Summary

![alt_text](https://github.com/DayuanTan/AITensorFlowSpecialization/raw/master/img/4convnetsummary.png)

- 298 = 300 - 1 - 1 because of 3 x 3 filter
- 149 = 298/2 because of 2 x 2 max pooling
- 147 = 149 - 1 - 1
- 73 = 147/2
- 71 = 73 - 1 - 1
- 35 = 71/2
- 78400 = 35 x 35 x 64. Original size is 300 x 300 x 64 = 5760000. 

------

### 1.4.4 Training the ConvNet with fit_generator

#### 1.4.4.1 Compile (Loss func + optimizer)

```python
#old:
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#new:
from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001), # lr is learning rate
              metrics=['acc'])
```

More about learning rate:[Here](https://www.youtube.com/watch?v=zLRB4oupj6g&feature=youtu.be)

#### 1.4.4.2 Training

```python
#old:
model.fit(training_images, training_labels, epochs=5)
#new:
history = model.fit_generator(
      train_generator, # which we setup earlier
      steps_per_epoch=8,  
      epochs=15,
      validation_data=validation_generator, # setup earlier
      validation_steps=8,
      verbose=2)
```

- **fit_generator()**: Call model.fit_generator instead of model.fit(), and that's because we're using a generator instead of datasets.
- **steps_per_epoch=8**: Since the batch_size=128 in train_generator earlier, and totally we have 1024 training images, so we need 8 batches to load all images.
- **validation_steps=8**: Since the batch_size=32 in validation_generator and we have 256 validation images totally, so we need 8 batches. 
- **verbose=2**: And the **verbose** parameter specifies how much to display while training is going on. With verbose set to 2, we'll get a little less animation hiding the epoch progress. 

#### 1.4.4.3 Prediction using this model once the model is trained

```python
import numpy as np
from google.colab import files
from keras.preprocessing import image

uploaded = files.upload()
```
So these parts are specific to **Colab**, they are what gives you the button that you can press to pick one or more images to upload. The image paths then get loaded into this list called uploaded

```python
for fn in uploaded.keys():
 
  # predicting images
  path = '/content/' + fn
  img = image.load_img(path, target_size=(300, 300)) # load an image
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0) # add dimension
```
The loop then iterates through all of the images in that collection. And you can load an image and prepare it to input into the model with this code. Take note to **ensure that the dimensions match the input dimensions** that you specified when designing the model

```python
  images = np.vstack([x])
  classes = model.predict(images, batch_size=10)
```
You can then call **model.predict**, passing it the details, and it will **return an array of classes**. In the case of binary classification, this will only contain one item with a value **close to 0** for one class and **close to 1** for the other (**sigmoid**).
```python
  print(classes[0])
  if classes[0]>0.5:
    print(fn + " is a human")
  else:
    print(fn + " is a horse")
```

------
A bit more:

 When you defined the model, you saw that you were using a new loss function called ‘[Binary Crossentropy](https://gombru.github.io/2018/05/23/cross_entropy_loss/)’, and a new [optimizer](https://www.tensorflow.org/api_docs/python/tf/train/RMSPropOptimizer) called [RMSProp](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf). If you want to learn more about the type of binary classification we are doing here, check out [this](https://www.youtube.com/watch?v=eqEc66RFY0I&t=6s) great video from Andrew!

 ------

 ### 1.4.5 Try it yourself (Without validation)

[Offical Code](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%208%20-%20Lesson%202%20-%20Notebook.ipynb)

 [My ccode](./myExercise/Horse_or_Human_NoValidation.ipynb)


 ### 1.4.6 Adding automatic validation to test accuracy

 How you can build validation into the training loop by specifying a set of validation images, and then have TensorFlow do the heavy lifting of measuring its effectiveness with that same. 

 [Official code](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%208%20-%20Lesson%203%20-%20Notebook.ipynb)

 [My code](./myExercise/Course_2_Part_2_Lesson_3_Notebook.ipynb)

#### Code: Without validation V.S. Add validation
```python
history = model.fit_generator(
      train_generator,
      steps_per_epoch=8,  
      epochs=15,
      verbose=1) #Without validation
```


```python
history = model.fit_generator(
      train_generator,
      steps_per_epoch=8,  
      epochs=15,
      verbose=1,
      validation_data = validation_generator,
      validation_steps=8) # Add validation
```

#### Output: Without validation V.S. Add validation
```python
Epoch 1/15
8/8 [==============================] - 9s 1s/step - loss: 0.9284 - acc: 0.5061
Epoch 2/15
8/8 [==============================] - 5s 609ms/step - loss: 0.7120 - acc: 0.6407
Epoch 3/15
8/8 [==============================] - 5s 674ms/step - loss: 0.4680 - acc: 0.8154
Epoch 4/15
8/8 [==============================] - 6s 781ms/step - loss: 0.8687 - acc: 0.8340
Epoch 5/15
8/8 [==============================] - 5s 588ms/step - loss: 0.3271 - acc: 0.8605
Epoch 6/15
8/8 [==============================] - 6s 770ms/step - loss: 0.1768 - acc: 0.9375
Epoch 7/15
8/8 [==============================] - 6s 691ms/step - loss: 0.0932 - acc: 0.9700
Epoch 8/15
8/8 [==============================] - 5s 681ms/step - loss: 0.2523 - acc: 0.8966
Epoch 9/15
8/8 [==============================] - 5s 675ms/step - loss: 0.0876 - acc: 0.9644
Epoch 10/15
8/8 [==============================] - 5s 679ms/step - loss: 0.1778 - acc: 0.9388
Epoch 11/15
8/8 [==============================] - 5s 684ms/step - loss: 0.3316 - acc: 0.8799
Epoch 12/15
8/8 [==============================] - 6s 787ms/step - loss: 0.0531 - acc: 0.9844
Epoch 13/15
8/8 [==============================] - 5s 592ms/step - loss: 0.0392 - acc: 0.9767
Epoch 14/15
8/8 [==============================] - 5s 681ms/step - loss: 0.0253 - acc: 0.9911
Epoch 15/15
8/8 [==============================] - 5s 680ms/step - loss: 0.0105 - acc: 0.9978
```


```python
Epoch 1/15
8/8 [==============================] - 8s 1s/step - loss: 0.8665 - acc: 0.5176 - val_loss: 0.6109 - val_acc: 0.7812
Epoch 2/15
8/8 [==============================] - 6s 770ms/step - loss: 0.7010 - acc: 0.6305 - val_loss: 0.5230 - val_acc: 0.7656
Epoch 3/15
8/8 [==============================] - 6s 744ms/step - loss: 0.5335 - acc: 0.7241 - val_loss: 2.6840 - val_acc: 0.5000
Epoch 4/15
8/8 [==============================] - 7s 835ms/step - loss: 0.7315 - acc: 0.8135 - val_loss: 0.9132 - val_acc: 0.8359
Epoch 5/15
8/8 [==============================] - 6s 742ms/step - loss: 0.2578 - acc: 0.8877 - val_loss: 0.9754 - val_acc: 0.8359
Epoch 6/15
8/8 [==============================] - 5s 646ms/step - loss: 0.4231 - acc: 0.8437 - val_loss: 2.2623 - val_acc: 0.6914
Epoch 7/15
8/8 [==============================] - 7s 834ms/step - loss: 0.1735 - acc: 0.9453 - val_loss: 1.1795 - val_acc: 0.8281
Epoch 8/15
8/8 [==============================] - 6s 743ms/step - loss: 0.2638 - acc: 0.9066 - val_loss: 0.9428 - val_acc: 0.8594
Epoch 9/15
8/8 [==============================] - 6s 733ms/step - loss: 0.1872 - acc: 0.9299 - val_loss: 1.4680 - val_acc: 0.7969
Epoch 10/15
8/8 [==============================] - 6s 727ms/step - loss: 1.0978 - acc: 0.8977 - val_loss: 0.5576 - val_acc: 0.6367
Epoch 11/15
8/8 [==============================] - 6s 776ms/step - loss: 0.2320 - acc: 0.9043 - val_loss: 1.7978 - val_acc: 0.7422
Epoch 12/15
8/8 [==============================] - 7s 835ms/step - loss: 0.0536 - acc: 0.9844 - val_loss: 1.7838 - val_acc: 0.8047
Epoch 13/15
8/8 [==============================] - 5s 668ms/step - loss: 0.0145 - acc: 0.9961 - val_loss: 1.9170 - val_acc: 0.8125
Epoch 14/15
8/8 [==============================] - 7s 830ms/step - loss: 0.6540 - acc: 0.8525 - val_loss: 0.2873 - val_acc: 0.8867
Epoch 15/15
8/8 [==============================] - 5s 646ms/step - loss: 0.1223 - acc: 0.9703 - val_loss: 1.0839 - val_acc: 0.8047
```

#### A bit more:

When tried resize all images to 150 x 150. The training times will improve, but that some classifications might be wrong! This is a great example of the importance of measuring your training data against a large validation set, inspecting where it got it wrong and seeing what you can do to fix it.

#### Over fitting
If your training data is close to 1.000 accuracy, but your validation data isn’t, what’s the risk here?

- You’re overfitting on your training data

------

### 1.4.7 Exercise 4 Happy or sad

A happy or sad dataset which contains 80 images, 40 happy and 40 sad. Create a convolutional neural network that trains to 100%. 

[Official code](./myExercise/Exercise4_Answer.ipynb)

[My code](./myExercise/Exercise4_Question.ipynb)







