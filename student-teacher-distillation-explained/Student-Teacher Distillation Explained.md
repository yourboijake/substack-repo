# Student-Teacher Distillation, Explained

Subtitle: From ResNet-18 to ViT on CIFAR-10, an explanation of Touvron et al, 2021

The technique known as "distillation" is a powerful way to train a "student" model from an already-trained "teacher" model, allowing for SOTA results to be achieved by smaller models with more data and compute efficiency [^1].

In this essay, I'll focus on the paper "Training data-efficient image transformers {\&} distillation through
attention" by Touvron et al [^2]. In this paper, the authors describe an approach for distillation of a convolutional neural net to a vision transformer. Doing so is useful because the "student" vision transformer model can be both smaller and is a more parallelizeable architecture, allowing for greater inference-time efficiency than the CNN "teacher". Before walking through the implementation in PyTorch, let's define what distillation training is, and try to conceptually understand why it's more efficient than ordinary training.

### Conceptual Background

In distillation training, we take an existing model, called the "teacher", which has been pre-trained on a large dataset, say ResNet for image classification on the ImageNet dataset. Training the teacher typically involves minimizing the cross-entropy loss, which only takes into account the model's predicted probability for the correct class on each training example. These are known as "hard" labels.

![Automobile class prediction example](auto_class_predictor.png "Class Predictions for Automobile")

In the figure above, imagine we have a model producing a discrete probability distributions across the 10 classes. The loss is computed as the negative log of the probability assigned to the correct class label, but the other 9 of these probabilities are not factored into the loss on this training example. And how could we use these other 9 probability values? We have no sense for the fact that some classes might be more "similar", in some sense, to one class than another. 

So, during pre-training, our best option is training against cross-entropy loss on the correct label, known as a "hard" label, which should eventually provide useful image representations to facilitate classification. However, once we have a model that has generated these useful representations, we can expect its outputs to produce a reasonable probability distribution over class labels. In fact, we do see this when using a pretrained model to predict the class of the automobile image show above [^3]. Here are the real outputs:

airplane: 10.87
automobile: 12.86
bird: 9.51
cat: 11.36
deer: 8.91
dog: 9.47
frog: 9.42
horse: 7.61
ship: 8.90
truck: 11.10 

It is desireable for the model to assign higher probability to car than to, say frog, as it has more "truckness" (wheels, headlights, placed on a street, etc.) than the frog does, even though it's not the correct class. Learning which general patterns correspond with "truckness" is a core part of image classification, and of generating useful convolution or embeddings (depending on what model architecture you choose to train). However, as we see from the distribution, the teacher isn't perfect, and it's distribution doesn't necessarily correspond with the distribution we might expect: for example, the model assigns higher probability to "cat" than "truck".

Where does this all leave us? Well, provided it has very good accuracy, the outputs of a pretrained model can be thought of as "soft" labels, in contrast to the "hard" labels used to train it. By training a model against the full output distribution of the teacher model, we can have a much richer set of gradients, informed by the full set of output probabilities. Using a formula to compute the distance between two probability distributions called <a href="https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence">Kullback-Leibler Divergence</a>, we can use the full soft label to train a classifier. To mitigate the impact of incorrect labels assigned by the teacher model, we'll apply a weighted average between the hard and soft labels.

### Technical Details

To demonstrate the efficacy of the student-teacher distillation approach in Touvon et al, 2021, let's implement the training for two vision transformer (ViT) models. To allow for an apples-to-apples comparison, I'll use the same model architecture and parameter count, and we'll compare the training speed. If the authors of the paper are correct, we expect that the student-teacher model will be able to improve its classification accuracy at a faster rate than normal cross-entropy loss.

So let's get started. First, let's use timm to import untrained tiny vit models and the ResNet-18 model for our teacher model, along with the CIFAR-10 dataset/data loader. We'll resize and normalize the dataset to accomodate our imported models.
```python
vit_normal = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=10)
vit_student = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=10)
resnet18 = timm.create_model("resnet18_cifar10", pretrained=True)

transform = transforms.Compose([
  transforms.Resize(224),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.49139968, 0.48215827 ,0.44653124],
      std=[0.24703233, 0.24348505, 0.26158768])
])

trainset = torchvision.datasets.CIFAR10(
  root='./data', train=True, download=True, transform=transform
)
testset = torchvision.datasets.CIFAR10(
  root='./data', train=False, download=True, transform=transform
)

trainloader = DataLoader(
  trainset, batch_size=batch_size,
  shuffle=True, num_workers=2)
testloader = DataLoader(
  testset, batch_size=batch_size,
  shuffle=False, num_workers=2)
```

Next, we'll define our training function, which optionally applies either normal cross-entropy (CE) loss on the hard labels or a mixture of CE loss and KL divergence. The weightings for the averaging of these two loss values is controlled by the hyperparameter alpha. Additionally, consistent with Touvon et al, 2021, we apply a "temperature" hyperparameter to smooth the soft label distribution.

```
def train(model, trainloader, testloader, learning_rate, weight_decay,
  num_epochs, teacher_model=None, temperature=0.5, alpha=0.5):
 
  optimizer = optim.AdamW(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
  ce_loss_criterion = nn.CrossEntropyLoss()
  kld_loss_criterion = nn.KLDivLoss(reduction='batchmean')
  scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

  train_epoch_losses, train_epoch_acc, test_epoch_acc = [], [], []
  epoch_values, iters, train_losses, train_acc = [], [], [], []

  for epoch in range(num_epochs):
    total_loss = 0.0
    correct = 0
    total = 0
    model.train()

    pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{num_epochs}')
    for batch_idx, (inputs, labels) in enumerate(pbar):
      #forward and backward pass
      outputs = model(inputs)
      if teacher_model:
        with torch.no_grad():
          soft_labels = F.softmax(teacher_model(inputs) / temperature, dim=1) #get soft labels from teacher model
        soft_preds = F.log_softmax(outputs / temperature, dim=1)
        kld_loss = kld_loss_criterion(soft_preds, soft_labels) * (temperature ** 2)
        ce_loss = ce_loss_criterion(outputs, labels)
        loss = alpha * kld_loss + (1 - alpha) * ce_loss #combine CELoss and KL-Divergence Loss
      else:
        loss = ce_loss_criterion(outputs, labels)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      #update running statistics
      total_loss += loss.item()
      preds = outputs.argmax(dim=-1)
      correct += preds.eq(labels).sum()
      total += labels.size(0)
 
      #update progress bar
      pbar.set_postfix({
        'Loss': f'{loss.item():.4f}',
        'Acc': f'{100.*correct/total:.2f}%'
      })
      epoch_values.append(epoch)
      iters.append(batch_idx)
      train_losses.append(loss.item())
      train_acc.append(correct / float(total))

    # update learning rate
    scheduler.step()

    # Calculate epoch metrics
    epoch_loss = total_loss / len(trainloader)
    epoch_acc = 100. * correct / total
    test_acc = eval_model(model, testloader, device)

    train_epoch_losses.append(epoch_loss)
    train_epoch_acc.append(epoch_acc)
    test_epoch_acc.append(test_acc)

    print(f'Epoch {epoch+1}: Train CELoss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, Test Acc: {test_acc:.2f}%')

  results = {
    'train_epoch_losses': train_epoch_losses,
    'train_epoch_acc': train_epoch_acc,
    'test_epoch_acc': test_epoch_acc,
    'epoch_values': epoch_values,
    'iters': iters,
    'train_losses': train_losses,
    'train_acc': train_acc
  }
  return results
```

Next, we'll use this training function to train train a ViT against the normal cross-entropy loss. Then, we train a ViT against ResNet-18's soft labels using an alpha-weighted average of KL divergence and cross-entropy loss.

```
WEIGHT_DECAY = 1e-2 #l2 regularization in cross-entropy loss
NUM_EPOCHS = 5

vit_normal_results = train(
  vit_normal,
  trainloader,
  testloader,
  device,
  learning_rate=learning_rate,
  weight_decay=WEIGHT_DECAY,
  num_epochs=NUM_EPOCHS)

vit_student_results = train(
  vit_normal,
  trainloader,
  testloader,
  learning_rate=student_learning_rate,
  weight_decay=WEIGHT_DECAY,
  num_epochs=NUM_EPOCHS,
  teacher_model=resnet_teacher,
  temperature=temperature,
  alpha=alpha)
```

### Training Results

Consistent with Touvron et al, the student-teacher approach with soft labels converges more quickly than the regular approach using hard labels. Here, we see the rate of improvement in training set accuracy over 5 epochs (the start of each epoch is marked by a grey dashed line).

![Training Set Accuracy by Batch](trainset_acc.png "Training Set Accuracy")

The vit_student slightly outpaces the rate of improvement of the vit_normal model. The gap becomes slightly more pronounced when we zoom in on the first epoch, which has the steepest rate of improvement:

![Training Set Accuracy by Batch, First Epoch](trainset_acc_epoch1.png "Training Set Accuracy, First Epoch")

The results are hardly dramatic on such a simple dataset, where hard labels provide rich enough information to learn query. However, there is a distinctly higher rate of accuracy improvement with the student-teacher model.

To see all the source code, visit the GitHub repo: https://github.com/yourboijake/substack-repo/tree/main/student-teacher-distillation-explained. Thanks for reading!


----------------------------------
[^1] Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. "Distilling the knowledge in a neural network." arXiv preprint arXiv:1503.02531 (2015).

[^2] Touvron, Hugo, et al. "Training data-efficient image transformers & distillation through attention." International conference on machine learning. PMLR, 2021.

[^3] Uses the "resnet34_cifar10" model in the timm library with pretrained weights.