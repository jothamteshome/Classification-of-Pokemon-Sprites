# Classification Of Pokemon Sprites

## 1. Introduction

<p>Pokemon has become a household name
over the past few decades, and with the franchise
nearing its 30th year, there are now over 1000
Pokemon in the game. Due to the franchise's long
history, many players that began playing in the earlier
generations and have since stopped may have trouble
recognizing newer Pokemon sprites. At the same
time, it can even be difficult for consistent players to
keep up with the ever-growing library of Pokemon
there are to choose from, which is only made harder
given the pixelated nature of the in-game sprites.</p>

<p>In this project, we aim to train a neural
network that can easily and accurately classify
Pokemon sprites found within a set of images. With
this, we hope to make the identification of Pokemon
less of a chore, and something that can be done
quickly through a machine learning model.</p>

<p>To do this, we implement a neural network,
utilizing the nature of this type of model to handle the
complexity that can be seen in classifying multiple
types of images into Pokemon. This model can be
given a screenshot of an image, and through the use 
of an image processing library, can isolate the sprite
of the rival Pokemon appearing on the screen. From
here, the model will be fed the isolated sprite image,
where it will finally be able to classify and return the
name of the Pokemon that corresponds to the given
sprite.</p>

## 2. Methodology
### 2.1. Data Collection
<p>To obtain the data necessary for training our
classification model, we are taking two approaches.
The first method of data collection is to use real battle
images from Pokemon games to collect the sprites of
Pokemon in a live setting. To do so, we plan to take
screenshots of the Pokemon battles with the use of
emulators. We will need to make sure the pokemon in
each image is clearly visible. We also would like to
include images taken from various orientations, so
that our model can account for this when classifying
images. To make these images suitable for our
purposes, we will also need to preprocess them,
which we will cover in the following section.</p>

<p>The second method for data collection that
we plan to use is data found across multiple sources
containing images of Pokemon and their sprites.
Using the various sources of images we found, we
will combine them to create one large dataset that we
can use to train and test our model performance. One
of the datasets we are using is Abdullah [1]. This
dataset contains over 1000 high-quality images of
Pokemon sprites, which will be useful to us as it
provides many sprites that we can use when
classifying Pokemon. Another dataset we are using is
Zhang [2]. While this dataset does not contain purely
sprite images of Pokemon, we still believe it will
provide us with useful images when classifying
Pokemon. Since the number of structured data sets is
limited, we will also augment images from data
sources and include them as additional samples. For
more information on this, please refer to section
2.2.1.</p>

<p>After images were collected and
augmentations were applied, we had 62 images for
each pokemon. We trained our model to classify 10
different pokemon, for a total of 620 different
images.</p>

### 2.2. Model Creation
#### 2.2.1. Data Preprocessing
<p>The preprocessing of our data is done in
several steps. First, it is important to note that many
of the source images are named as “x.png” where x is
a unique id assigned to each pokemon. With that
being said, we needed to establish an association
between each pokemon’s ID and their name. A data
set called “pokedex” solved this problem with a csv
file. The row number of the file matched the
pokemon’s id and included other information such as
the pokemon’s name and type. We extracted this
file’s information as a pandas dataframe and utilized
it to put source images into the correctly named
pokemon sub folder.</p>

<p>For each pokemon ID within the range [0,
50), we search the source directories for images
named “[ID].png”. If a match is found, it is placed in
a subdirectory of the pokemon’s name along with all
the other matched images. In addition to placing the
original source images, we also created augmented
samples. These samples were either rotated by a
random amount, zoomed in slightly, or gaussian noise
was applied. The augmented images were placed in
the same named subdirectory as the original source
images.</p>

<p>We included augmented images in our
sample set for two reasons. First, adjusted images
served as another source of input images, allowing us
to establish a large training data set. This was
important because there were only a few organized
pokemon data sets and this method ensured we would
have enough samples. We also included modified
images because pokemon in battles are not
guaranteed to be static. Each pokemon has a different
move set, even within its own species, that have the
potential to temporarily modify the appearance of the
opponent or itself. For example, in Figure 1 below,
the pokemon Shaymin is hit with a psychic power,
causing its appearance to temporarily become
distorted.</p>

<p align="center"><img src="https://github.com/jothamteshome/Classification-of-Pokemon-Sprites/assets/94427010/d8180b2a-221e-473c-922f-671e66cc8069"/></p>
<p align="center">Figure 1. Shaymin before(left) and after (right) being attacked by a psychic power</p>
<br/>

<p>In another instance, the pokemon Milotic can flip
between two different orientations, as showcased in
Figure 2.</p>

<br/>

<p align="center"><img src="https://github.com/jothamteshome/Classification-of-Pokemon-Sprites/assets/94427010/356ded8c-e49e-429c-8945-28f93f1936d3"/></p>
<p align="center">Figure 2. Milotic appearance as it enters battle (left) and as it selects its first move (right)</p>
<br/>

<p>In each case, the augmentations we have
made increase the robustness of our model. Adding
noise allows us to better classify instances such as the
Shaymin example and rotating images allow us to
better classify variations as showcased through the
Milotec example. Since pokemon are often reused
across games, image variations allow our model to
better classify modifications that are made in newer
games.</p>

#### 2.2.2. Model Architecture
<p>For our convolutional neural network, we
used PyTorch to design and create the architecture.
We decided to take a simple approach for our initial
model and plan on making the model more complex
based on our initial results. Our initial model consists
of two 2D convolutional layers, with kernel size (3,
3), that are each followed by a Rectified Linear Unit
(ReLU) layer. The ReLU layers in our network
introduce non-linearity to our model and help it learn
complex patterns in the data. Across these 2D
convolutional layers, we increased the number of
channels from 3 to 64, then from 64 to 128. By
increasing the number of channels, it allowed for our
model to learn and capture a more diverse set of
features from the input sprite images the model was
trained on. Next, we used a 2D max pooling layer to
downsample the spatial dimensions of the input data
while retaining the most important information.
Finally, we flattened the features and used a series of
two linear layers and a ReLU layer to map the
features into the number of classes we are trying to
predict from. This initial model architecture can be
seen in Figure 3 below.</p>

<p align="center"><img src="https://github.com/jothamteshome/Classification-of-Pokemon-Sprites/assets/94427010/0131cd5e-1af4-4c90-9453-1c280664abe8"/></p>
<p align="center">Figure 3. Initial model architecture</p>
<br/>

<p>After seeing the results of the initial model
model, we decided it would be best to modify this
model to increase the model’s ability to classify the
Pokemon sprites. First, dropout regularization layers
were added to the network to help prevent overfitting.
The next step was adding 2D batch normalization
layers. This was done to help stabilize the training
process. Finally, the size of the output dimension of
the second convolutional layer and ReLU block was
increased from 128 channels to 256, which was done
to capture even more features from the data than the
initial model. The full architecture of the final model
can be seen in Figure 4.</p>
<br/>
<p align="center"><img src="https://github.com/jothamteshome/Classification-of-Pokemon-Sprites/assets/94427010/3b4dd02b-893e-499f-a204-502d48c423d6"/></p>
<p align="center">Figure 4. Final model architecture</p>
<br/>

#### 2.2.3. Model Training
<p>To train our initial model, we use the
preprocessed sprite images mentioned in Section
2.2.1. Determining the best hyperparameters using
these images was also a difficult process, as there
were many factors at play. Initially, we had trained
the model using SGD as our optimizer, though we
had also experimented using RMSprop as well as the
Adam optimizer, all of which were available through
Pytorch. Through experimentation though, we found
that SGD was the optimizer that allowed our model
to best classify our test images, so it was the
optimizer we chose when training the initial model.</p>

<p>Knowing which optimizer we planned to
use, the next step was to determine the best values for
the rest of our hyperparameters. Naturally, the
learning rate was a good place to start. Initially, we
had run into a problem where we could train the
model, but no matter how many epochs we trained
for, we were consistently seeing a training loss of
‘nan’ at every epoch. What we found after some trial
and error was that our model was unable to converge
when using our initial learning rate of 1e-3. We did
find that using a learning rate of 1e-4 mitigated this
issue, so we decided to use this as our initial learning
rate and experiment from there. Overall, we tested
learning rate values on the interval of [5e-5, 9e-4],
initially increasing by 1e-5, then increasing by 1e-4
one we reached a learning rate of 1e-4 and greater.
From this testing, we found that the best learning rate
for our purposes was 4e-4, resulting in a validation
accuracy of greater than 0.73, our best accuracy to
that point.</p>

<p>Once we had our learning rate, the final
hyperparameter we extensively experimented with
was the number of epochs to train for. Initially we
had decided on 10 epochs, but we felt that testing a
wide range of values could result in better evaluation
metrics. Thus, we tested between [10, 50] epochs,
and found that for our initial model, we were able to
achieve similar accuracy with anywhere from 25 to
50 epochs.</p>

<p>With all of our testing done, we finally
settled on the hyperparameters we would use to train
our initial model. These hyperparameters included
training our model for 25 epochs using the SGD
optimizer, a learning rate of 4e-4, a weight decay of
1e-4, a momentum of 0.9, and batch size of 16, using
a total of 2154 images to train our model.</p>

<p>After we had designed our final model, we
had to perform some hyperparameter tuning for the
model as well. We began with the initial model’s
hyperparameters, as we felt it was a good starting
point rather than beginning from scratch. We also ran
through a similar method for tuning, first determining
the right optimizer, the learning rate, and finally the
right number of epochs for our model. What we
found through our testing was that the
hyperparameters we had begun our tuning with had
actually performed the best for both the initial model
as well as the larger final model.</p>


#### 2.2.4. Model Results
<p>To test our initial model, we use a subset of
the sprites that we have preprocessed. This subset of
images was generated and set aside prior to the
training of the model, so these specific images have
not been seen by the model before. Once the images
are prepared in a way that our model will accept, we
feed these images into the model to evaluate our
results. Currently, we are seeing an accuracy of
0.737, which we believe is a good start for our model
considering the number of classes we are assigning
images to as well as the simple architecture we are
using for our current model. We are also testing for
metrics such as precision, recall, and f1-score. For
these, we are also seeing good results. To test for
these values, we use sklearn’s precision_score,
recall_score, and f1_score functions, using these on
the ‘weighted’ mode to account for a potential
imbalance due to the labels we have. Using these, we
are able to see that our model is achieving a precision
of 0.758, a recall of 0.737, and an f1-score of 0.737.
These values can be seen in Figure 5 below. Overall,
these are good values to be seeing on our initial
model, and we believe we will see an even greater
improvement on these metrics as we continue to
improve our model.</p>

<p>After obtaining our original results, we
continued to experiment making our convolutional
neural network more complex, varying the
preprocessing of our sprite images, and tuning
hyperparameters. After this experimentation, we were
able to greatly improve the classification results of
our model. Preparing the training and test sets the
same way as mentioned above, we were able to
achieve an accuracy of 0.986, which is significantly
higher than our initial results. Adding dropout and
batch normalization errors greatly helped our model
reduce overfitting that was being seen prior.
Additionally, our model produced a precision of
0.987, recall of 0.986, and an f1 score of 0.986. These
are also great metrics to see. These results can also be
seen in Figure 5 shown below</p>

<p align="center"><img src="https://github.com/jothamteshome/Classification-of-Pokemon-Sprites/assets/94427010/24e25f73-ba6d-4e13-986e-92abeb1df103"/></p>
<p align="center">Figure 5. Results Classifying Sprites</p>
<br/>

<p>While our final results ended up being very
good, we still wanted to see the breakdown between
the pokemon classes. We did not want to include all
of the first 50 pokemon, since the results are very
similar and would take up unnecessary space, so we
used a function from sklearn called
classification_report to show the breakdown of the
first 10 pokemon, which can be seen below in Figure
6. While our model performed very well on all the
pokemon, it is worth noting that some of the lower
scores could be due to some pokemon looking similar
to their evolutions.</p>

<p align="center"><img src="https://github.com/jothamteshome/Classification-of-Pokemon-Sprites/assets/94427010/bdcb7e0f-080f-49ee-9574-0ccaf059da5c"/></p>
<p align="center">Figure 6. Classification Results of First 10 Pokemon</p>
<br/>

## 3. Battle Image Processing
<p>When it comes to finding objects on images
there are multiple ways to approach this problem, it
could be using the colors of the pixels as references,
processing the image to get the edges from it and try
to find a desired shape, and so on. We decided to go
with the approach of finding a shape within the image
to get the rival pokemon from a screenshot of the
game.</p>

### 3.1. Finding Shapes on Images
<p>When it comes to pokemon sprites is really
hard to specify a single shape that can identify any
pokemon, specially if we consider the full picture,
because we are on the scenario of a pokemon battle
there will always be at least 2 pokemon on the image,
so to reduce the complexity of this task we are only
going to consider 1vs1 battle and limit the search on
the right side of the image.</p>

<p>To be able to complete this task we are using
OpenCV to cut the image and keep only the right
side, then using the function GaussianBlur(...) we
smooth the image, to finally use the Canny(...) and
findContours(...) functions to find the edges of the
image and the shapes that these are part of. Figure 7
is the original image from a pokemon battle, and
Figure 8 represents how these transformations look
like on the previous image.</p>

<p align="center"><img src="https://github.com/jothamteshome/Classification-of-Pokemon-Sprites/assets/94427010/6b504050-2062-421d-87a5-34759fd1ef1e"/></p>
<p align="center">Figure 7. Original image</p>
<br/>

<p align="center"><img src="https://github.com/jothamteshome/Classification-of-Pokemon-Sprites/assets/94427010/c2f74f60-474c-493e-a4f7-93cfbbf3db1e"/></p>
<p align="center">Figure 8. Edges from a pokemon battle image</p>
<br/>

### 3.2. Selecting a Shape
<p>As said before selecting a single shape that
represents all the pokemon is really hard if not
impossible, so the approach that we decided to take to
overcome this issue is to take advantage of the simple
interface that the pokemon games use, having most of
its components being rectangular we can estimate
that if we take the most complex shape from the
image it is most likely to be the pokemon or at least a
segment of the image where this is contained or close
to. So following this line of thought we find the most
complex shape on the image by listing all the shapes
using the OpenCV function findContours(...) and
then find which shape is the one with the most edges
connected to it, then by knowing the position of the
shape we isolated at least one third of the image using
its position as the center. Figure 9 represents the final
result after applying this process.</p>

<p align="center"><img src="https://github.com/jothamteshome/Classification-of-Pokemon-Sprites/assets/94427010/78761710-a1fc-4bdf-9578-2021db556c68"/></p>
<p align="center">Figure 9. Isolated image of the most complex shape</p>
<br/>

### 3.3. Results
<p>After locating the part of the image where
the rival pokemon is, what is left to do is just isolate
the portion of the original image in the same position
as the edges representation. Figure 10 shows the final
result.</p>

<p align="center"><img src="https://github.com/jothamteshome/Classification-of-Pokemon-Sprites/assets/94427010/b012dfa7-ff53-4f9e-a079-657246d876bd"/></p>
<p align="center">Figure 10. Isolated portion of the original image</p>
<br/>

<p>The main problem with this approach is that
sometimes the most complex shape on the images is
not contained inside the pokemon, it can be one letter
from the name or a number from the stats, an
example of this is on Figure 11 and Figure 12.
Reduce the impact from this problem, instead of
getting just one image we take the top 5 shapes to get
5 different images from each battle.</p>

<p align="center"><img src="https://github.com/jothamteshome/Classification-of-Pokemon-Sprites/assets/94427010/8ff41ea5-db4f-4073-afd6-40ec0b7d3efc"/></p>
<p align="center">Figure 11. Original Battle image</p>
<br/>

<p align="center"><img src="https://github.com/jothamteshome/Classification-of-Pokemon-Sprites/assets/94427010/b4fefbcf-1346-4fe6-b973-caf6bf4b78a5"/></p>
<p align="center">Figure 12. Isolated portion without a pokemon</p>
<br/>

### 3.4. Model Results with Battle Images
<p>Due to the automated battle image
processing, which cropped the battle images based on
the most complex shapes in the image, not always
cropping the attacking pokemon in the battle
completely in frame, we decided to compare our
classification results between the automatically
cropped battle images with the classification results
from manually cropped battle images.</p>

<p>Due to inconsistencies in our results, we
were able to receive more consistent results when
dealing with only the first 10 pokemon, instead of the
first 50. To accomplish this, we trained our model on
only the first 10 pokemon sprites and used this model
to test on these battle images.</p>

#### 3.4.1. Battle Images Cropped Based on Shapes
<p>After automatically cropping the battle
images, we were able to obtain the accuracy,
precision, recall, and f1 score for varying Top-K
classifications. We decided to use Top-K
classifications since we were obtaining low results
for single-label classification. As expected, as K got
larger so did our evaluation metrics. Our results can
be seen below in Figure 13. It is worth noting that our
results are only slightly better than random. This may
be due to a lot of reasons, but one reason could be the
inconsistent results from our automated cropping,
which does not always crop the pokemon correctly.
The slightly better than random results indicate that
our classifier was able to correctly classify some of
the pokemon better than chance.</p>

<p align="center"><img src="https://github.com/jothamteshome/Classification-of-Pokemon-Sprites/assets/94427010/a37889a9-a568-4e98-8e36-aee49cf67bf0"/></p>
<p align="center">Figure 13. Results Classifying Battle Images Automatically Cropped</p>
<br/>

#### 3.4.2. Battle Images Manually Cropped
<p>As mentioned previously, we decided to
manually crop these battle images to see if our
trained convolutional neural network was able to
correctly classify these images at a rate better than we
saw with our automated cropping. After testing on
these manually cropped images with our trained
model, we found significantly better results. We
found an accuracy of 0.500 with single-label
classification and an accuracy of 0.857 with Top-5
classification. Both these results are greatly above
random chance. We also found that the precision,
recall, and f1 scores also increased as well. These
results can be seen in Figure 14 below.</p>

<p align="center"><img src="https://github.com/jothamteshome/Classification-of-Pokemon-Sprites/assets/94427010/618ea742-7657-4fd5-87e7-dd0477fcb3b1"/></p>
<p align="center">Figure 14. Results Classifying Battle Images Manually Cropped</p>
<br/>

## 4. Conclusions
<p>By taking the challenge on detecting
pokemons on any screenshot from a game we were
able to notice that if the pokemon sprite is properly
isolated it is possible to have a decent chance to
detect which pokemon it is without looking at any
property besides its sprite.</p>

<p>The main problem comes when there isn’t
any specific rule to know where a pokemon is going
to be located at, besides that is going to be on the
right side of the image, so it is necessary to find a
way to get an approximate location, in this case we
used an approach of getting the location of the
pokemon by finding the most significant shapes on
the image, and even if this didn’t return the best
results within the model it is possible to tell that the
most likely reasons for these miss classifications
were that there wasn’t enough features from the
pokemons to get recognized by the model and that
this could be improve by manually isolating the
pokemon.</p>

## 5. Future Work
<p>Our model performs well when classifying a
small subset of pokemon with minor variations
applied. With that being said, the world of Pokemon
is expansive and continues to evolve every year. We
believe that our work could be built upon to create a
flexible, generation-independent model.</p>

<p>One limitation to our model’s performance
was the inconsistencies that occurred when retrieving
pokemon from battle images. If we were to continue
to work on our model, we would strive to discover a
more consistent method of processing battle images
so that our testing data would hold more similarity to
the training data.</p>

<p>Since our model only trained using
Pokemon generation 1 images, future researchers
could test whether our methodology applies to other
Pokemon generations as well. We believe that if our
work can be applied to other pokemon generations, it
will be possible to create a classifier that detects a
large number of Pokemon across a wide range of
generations. Though, if a multi-generational classifier
is not feasible, it may be of interest to determine
whether it is possible to classify which Pokemon
generation an image comes from. If this were
possible, pokemon could technically be classified
using two models: one to determine the
generation/source of the photo and then another to
determine the pokemon from that generation/source.</p>

<p>Finally, our model utilized sprite and game
battle data, though there are other mediums through
which Pokemon can be found. TV shows, movies,
and many drawings have been made from the
Pokemon franchise, and it would be interesting to see
whether a classifier could determine the difference
between Pokemon with training data from these
outlets.</p>

## References
* [1] Wasiq Abdullah. PokeVision 1010, 2023.
* [2] Lance Zhang. 7000 Labeled Pokemon, 2019.
