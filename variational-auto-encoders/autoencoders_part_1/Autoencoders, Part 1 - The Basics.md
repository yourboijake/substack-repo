As companies like Surge AI and Scale AI demonstrate, there is a huge challenge involved in generating datasets for supervised learning. I saw this firsthand at my first job, working on an ML team assembling datasets to train fraud and spam call detection models for big telcos. Generating high-quality data annotations is super expensive and error-prone, which bottlenecks ML approaches reliant on these kinds of datasets.

Over the next two (or more) essays, I want to explore a family of ML techniques used to generate compressed data representations ("embeddings") which are both useful for transfer learning to other ML contexts, and don't require labelling. These models are called Autoencoders.

#### Background

At a high level, an autoencoder is a model that is trained to compress its dataset into a lower-dimensional embedding (the "encoder"), then reconstruct the original input from this embedding (the "decoder"). This encoder-decoder pair is collectively referred to as an autoencoder. The graphic below, from Dor Bank, Noam Koenigstein, and Raja Giryes illustrates the autoencoder nicely:

![[Pasted image 20250726095143.png]]

The motivation behind the use of this approach relies fundamentally on the concept of an embedding, and what makes them useful. In the case of the image above, we start with an image of a six. Taken from a dataset like MNIST, this image is initially represented as a grid of pixel values. For MNIST, these images are 28 x 28 pixels, so each image can be thought of as a vector of 784 floats. This 784-element vector is a poor representation (a poor embedding, we could say) of handwritten numbers for two reasons: 
1. High dimensionality: it's very high-dimensional relative to its information content. Many, if not most, of the pixel values are redundant, and it is possible to represent the data in a much more compressed way, reducing the compute needed to process inputs.
2. Not useful for separating out classes: this representation of, say, the number six doesn't do a good job of making all sixes in the MNIST dataset similar to each other, and representing other digits differently. For example, let's apply cosine similarity (a commonly used metric to compare the similarity of two vectors) to the pixel value vectors of a six from the MNIST dataset against another six, an eight, and the original six, but shifted 3 pixels to the left.
![[cos similarity on sixes.png]]

Here we see that the original six is more similar to another six than to the eight, but is even more similar to an eight than to the exact same image, simply shifted 3 pixels left! As a representation of the content "six", the pixel-level image vector lacks almost any semantically meaningful information.

To overcome these limitations of high dimensionality and semantically meaningless embeddings, we can apply some sort of machine-learned transformation to the raw pixel data, projecting it into a lower-dimensional, useful representation of the image. One way to learn this transformation is through supervised learning (given an image of a six, predict correct label). Another approach, which doesn't require labels, is autoencoding.

#### Autoencoders

Fundamentally, an autoencoder is quite simple: we transform a high-dimensional input like an image or a chunk of text into a lower-dimensional embedding vector space. Then, we project this back into the original input dimensionality, and we compare our reconstructed data to the original. The lower-dimensional vector representation creates a sort of "information bottleneck", forcing the model to learn transformations that capture all the relevant information for distinguishing different images in the training set.

To demonstrate, I trained an autoencoder on the MNIST dataset using a ResNet-style architecture Github link: git repo link here). Below are the reconstructed examples for several images:

![[img reconstructions.png]]
The top image is the input, from MNIST, and the bottom row is the reconstructed image from the embedding through the decoder. The original MNIST images are 28 x 28, and the embedding space used here 32-dimensions. Note, this is a compression factor of 24.5 (28 * 28 = 784, down to 32 dimensions). 

In the reconstructions we can observe some of the important properties of autoencoders and embeddings. Three I'd like to highlight are:
1. **The blurring of the images:** the reconstructed images are blurrier than the inputs, due to the fact that compressing the images so aggressively (by a factor of 24.5) results in a loss of information. Although the reconstructions are recognizable, they have lost some of the detail from the original input images.
2. **The "cleanup" of the input image:** the reconstructed outputs lack some of the peculiarities of the inputs. For example, the reconstructed 3 looks more "generically" like a 3 than the input. Similarly, the stem on the 9 isn't very straight, but gets straightened out in the reconstructed 9. These are because the autoencoder must learn the general characteristics of digits to reconstruct them, and quirks of any individual image get lost in this process due to the low dimensionality of the embedding bottleneck.
3. **The five becomes a six**: the original image of the five is sloppy, and has a six-like loop at the bottom. The autoencoder, seeing this general visual structure, interprets it as the sort of pattern you'd see in a six, so the reconstruction looks much more like a 6 than a 5. This demonstrates the capacity and tendency of the autoencoder to interpret specific shapes in the context of larger categories, and automatically infer the category as a result.

#### Validating the Embeddings

One interesting thing that stood out to me after training these models is the ability of the autoencoder to learn class distinctions, *even without seeing any labels*. The model, in effect, learns many of the internal latent representations that a supervised classifier might despite having no conception of number, or even the fact that there are 10 classes in MNIST (integers 0-9).

To demonstrate the inferred, and totally unsupervised class separation/clustering, here is the cosine similarity (renormalized to be in range [0, 1]) of the autoencoder's embeddings for a random sample from the test set (withheld during training). To produce these values, I trained the autoencoder, then used it to generate embeddings for the held-out test set sample (n=1000). Then, I compared the cosine similarity of all these embeddings, resulting in a 1000 x 1000 matrix. Using this matrix, I compared the average cosine similarity within class vs between different classes, for each of the images in each of the ten classes. So, for example, we compare how similar the embeddings are for all the zeros compared to all the other zeros, and how similar the zeros are to all the non-zeros in the sample. The results for an untrained autoencoder, randomly initialized:

```
class    |  same class  |  diff class  |  ratio (same / diff)
-------------------------------------------------------------
class 0        0.69           0.69          1.01
class 1        0.69           0.68          1.02
class 2        0.70           0.70          1.01
class 3        0.72           0.71          1.02
class 4        0.73           0.71          1.03
class 5        0.72           0.70          1.02
class 6        0.72           0.70          1.02
class 7        0.69           0.69          1.00
class 8        0.73           0.71          1.04
class 9        0.72           0.70          1.02
```

The key metric here is the ratio (same / diff). This shows the relative average similarity of images in the same class compared to different classes. A ratio near 1 means that there is no differentiation between the same vs different classes, and ratio values significantly greater than 1 means the embeddings are differentiating classes. Here are the results after training, again, using a training process that makes *no use* of the labels at all:

```
class    |  same class  |  diff class  |  ratio (same / diff)
-------------------------------------------------------------
class 0        0.69           0.55          1.26
class 1        0.75           0.53          1.42
class 2        0.68           0.56          1.21
class 3        0.67           0.55          1.20
class 4        0.67           0.56          1.19
class 5        0.63           0.55          1.15
class 6        0.68           0.56          1.23
class 7        0.67           0.55          1.22
class 8        0.65           0.56          1.15
class 9        0.69           0.56          1.22
```

The ratio increases massively, as the embedding similarity within the same class increases relative to different classes. In other words, the autoencoder has learned to separate out classes despite having no conception of the class labels themselves, or even the number of different classes in the first place!

#### Conclusion

Writing this essay, I got a glimpse of the power of autoencoders to generate meaningful embeddings in a totally unsupervised way. If you're interested in the implementation details, and the results of the analytic methods discussed in this essay applied to the CIFAR10 dataset, see the Github repo: github repo link. I intend to follow this essay up with a discussion of variational autoencoders (VAEs), and the role they play in generative image and video models, so stay tuned.

#### Appendix

##### Results on CIFAR10
Applying the same autoencoder model architecture and cosine similarity comparison method to CIFAR10, here's the untrained model:
```
class    |  same class  |  diff class  |  ratio (same / diff)
-------------------------------------------------------------
class 0        0.49           0.50          0.99
class 1        0.52           0.49          1.06
class 2        0.49           0.50          0.99
class 3        0.59           0.54          1.10
class 4        0.61           0.54          1.13
class 5        0.58           0.53          1.08
class 6        0.52           0.50          1.04
class 7        0.53           0.52          1.02
class 8        0.60           0.53          1.14
class 9        0.58           0.53          1.09
```

And here is the trained model:
```
class    |  same class  |  diff class  |  ratio (same / diff)
-------------------------------------------------------------
class 0        0.16           0.04          3.63
class 1        0.19           0.08          2.31
class 2        0.10           0.07          1.36
class 3        0.20           0.07          2.68
class 4        0.14           0.08          1.70
class 5        0.13           0.07          1.95
class 6        0.35           0.08          4.55
class 7        0.10           0.06          1.78
class 8        0.25           0.02          15.30
class 9        0.28           0.06          5.06
```

Amazingly, the results for CIFAR10 are much more stark, and the ability to differentiate ships (class 8 in CIFAR10) increases by over a factor of 15.