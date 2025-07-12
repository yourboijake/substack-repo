Subtitle: From ResNet-18 to ViT on CIFAR-10, an explanation of Touvron et al

The technique known as "distillation" is a powerful way to train a "student" model from an already-trained "teacher" model, allowing for SOTA results to be achieved by smaller models with more data and compute efficiency. Footnote to Hinton, Vinyals, Dean Distilling the Knowledge in a Neural Network

In this essay, I'll focus on the paper "paper name" by Touvron et al. In this paper, the authors describe an approach for distillation of a convolutional neural net to a vision transformer. Doing so is useful because the "student" vision transformer model can be both smaller and is a more parallelizeable architecture, allowing for greater inference-time efficiency than the CNN "teacher". Before walking through the implementation in PyTorch, let's define what distillation training is, and try to conceptually understand why it's more efficient than ordinary training.

##### Conceptual Background

In distillation training, we take an existing model, called the "teacher", which has been pre-trained on a large dataset, say ResNet for image classification on the ImageNet dataset. Training the teacher typically involves minimizing the cross-entropy loss, which only takes into account the model's predicted probability for the correct class on each training example. These are known as "hard" labels.
Diagram here with image and predicted probabilities, highlighting the correct probability.
In the figure above, the model produces a discrete probability distributions across the 10 classes. The loss is computed as the negative log of the probability assigned to the correct class label, but the other 9 of these probabilities are not factored into the loss on this training example. And how could we use these other 9 probability values? We have no sense for the fact that some classes might be more "similar", in some sense, to one class than another. 

So, during pre-training, our best option is training against cross-entropy loss on the correct label, known as a "hard" label, which should eventually provide useful image representations to facilitate classification. However, once we have a model that has generated these useful representations, we can expect its outputs to produce a reasonable probability distribution over class labels. In fact, we do see this, as shown in the example below:

Example with image of truck, car, then frog. 
It is desireable for the model to assign higher probability to car than to frog, as it has more "truckness" (wheels, headlights, placed on a street, etc.) than the frog does, even though it's not the correct class. Learning which general patterns correspond with "truckness" is a core part of image classification, and of generating useful convolution or embeddings (depending on what model architecture you choose to train).

Where does this all leave us? Well, provided it has very good accuracy, the outputs of a pretrained model can be thought of as "soft" labels, in contrast to the "hard" labels used to train it. By training a model against the full output distribution of the teacher model, we can have a much richer set of gradients, informed by the full set of output probabilities. Usinga formula to compute the distance between two probability distributions called Kullback-Liebler Divergence, we can use the full soft label to train a classifier.

Diagram showing hard labels with cross entropy loss vs soft labels with KL Divergence loss

##### Technical Details

Given a set of soft labels from a pretrained teacher model, distillation seeks to minimize the discrete Kullback-Leibler divergence between the models' output probabilities. 
Image from Wikipedia with equation.

First, let's import the ResNet-18 model from PyTorch, along with the CIFAR-10 dataset/data loader.

Next, we'll implement our model, a simplified version of the ViT architecture (paper citation). As in ViT, we use the Vaswani et al architecture in "Attention Is All You Need", but with linear projections of image patches into an embedding space, rather than a token-vector embeddings.

Finally, we'll implement two training loops. In the first training loop, we train a ViT against the normal cross-entropy loss. In the second, we train a ViT against ResNet-18's soft labels using KL divergence.

Below are the results for the training curves and data. Consistent with Touvron et al, the student-teacher approach with soft labels converges more quickly than the regular approach using hard labels.

To see all the source code, link to GitHub repo. Thanks for reading!


Note: Touvron et al also introduce a "hard" KL divergence approach, which is not covered in this essay.
