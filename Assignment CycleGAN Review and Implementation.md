## Homework Assignment of Unit 1: CycleGAN Review and Implementation

**Due Date**: 11:59pm September 25 Wednesday   
**Submission**: Please submit your solution as a Jupyter Notebook (*.ipynb). Ensure your notebook runs without errors when executed from top to bottom. Please include clear comments and discussion if required, and print outputs as needed.

### Overview

In this assignment, you will review the CycleGAN model, a type of Generative Adversarial Network (GAN) used for image-to-image translation, where the goal is to transform images from one domain to another. The key idea is to learn bidirectional mappings between two different domains (e.g., transforming a picture of a horse into a zebra and vice versa). The task will focus on understanding the techniques behind CycleGAN, exploring its architecture, and implementing key components of the model. By the end of this assignment, you will have a deeper understanding of representation learning, generative models, and the practical skills to work with CycleGAN.

<img width="837" alt="Screenshot 2024-09-16 at 6 19 49 AM" src="https://github.com/user-attachments/assets/f48bde41-aa9d-4dbc-a109-2a995bdaf779">


### Background

**CycleGAN** (Zhu et al. 2017, https://ieeexplore.ieee.org/document/8237506) is a type of GAN that can learn mappings between two domains. Unlike Standard GANs, which is designed to generate new data samples that resemble a target distribution (e.g., generating realistic images from noise), CycleGAN learns from two set of images, which are not required to be paired.  For example, to learn to convert horses to zebras, CycleGAN does not require paired horse and zebra images in the training process; instead, it can learn from independent sets of horse images and zebra images, using cycle consistency to enforce the relationship between the two domains. 

The Figure below compares the framework of GAN and CycleGAN. Details of CycleGAN can be found from the paper: "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks" by Zhu et al. 2017, https://ieeexplore.ieee.org/document/8237506.  This framework has enabled versatile applications like style transfer, domain adaptation, and more, and this paper has been cited over 24,000 times until September 2024. 

![Picture1](https://github.com/user-attachments/assets/5d081ac6-f92b-421e-9833-432259c4a4c2)


### Assignment Tasks

#### Part 1: Understanding  CycleGAN (15 points)

**The structure of CycleGAN**: Read the CycleGAN paper, understand the basic principles of CycleGAN, and its architecture (including the generator, discriminator, and the loss functions). Answer the following questions:
   - What is the idea of designing a "cycle" in the image translation framework? Please describe the workflow by your own language  (9 points)
   - The full objective function in Equation (3) consists of several loss functions. Please explain how each of them contributes to the image translation from one domain to the other? (6 points)

 


#### Part 2: Implementation of CycleGAN (40 points)

**Task Description**:  
Implement the CycleGAN model, including the generator and discriminator models, and the adversarial and cycle consistency loss functions. The goal is to understand how the CycleGAN framework is constructed and how the key components interact within the model.

**Task**:  
Implement the CycleGAN model by completing the following steps:

1. **Define the Generator and Discriminator Models** (15 points):
   - Implement a simple `Generator` model using convolutional layers, normalization, and activation functions to translate images from one domain to another.
   - Implement a `Discriminator` model using convolutional layers and activation functions to classify images as real or fake.

2. **Implement Loss Functions** (15 points):
   - **Adversarial Loss**: Implement the adversarial loss function that helps the discriminator distinguish between real and fake images.
   - **Cycle Consistency Loss**: Implement the cycle consistency loss function that ensures images translated to another domain and back remain unchanged.

3. **Integrate Models and Losses into the CycleGAN Framework** (10 points):
   - Define the `CycleGAN` class that combines the two generators (one for each direction) and the two discriminators.
   - Implement the forward pass that integrates the generators and discriminators and computes the outputs required for loss calculation.

**Code Skeleton**:
The following code skeleton is provided for you to implement CycleGAN. Please fill in the necessary details for each model and function:

```python
import torch
import torch.nn as nn

# Define the Generator model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Define the layers of the generator (e.g., convolutional layers, normalization, activations)
        pass

    def forward(self, x):
        # Define the forward pass of the generator
        pass

# Define the Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Define the layers of the discriminator
        pass

    def forward(self, x):
        # Define the forward pass of the discriminator
        pass

# Define Adversarial Loss
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True):
        super(GANLoss, self).__init__()
        self.loss = nn.MSELoss() if use_lsgan else nn.BCEWithLogitsLoss()
    
    def forward(self, prediction, target_is_real):
        target_tensor = torch.ones_like(prediction) if target_is_real else torch.zeros_like(prediction)
        loss = self.loss(prediction, target_tensor)
        return loss

# Define Cycle Consistency Loss
def cycle_consistency_loss(real_image, reconstructed_image, lambda_weight=10.0):
    loss = nn.L1Loss()(reconstructed_image, real_image)
    return lambda_weight * loss

# Define the full CycleGAN model
class CycleGAN(nn.Module):
    def __init__(self):
        super(CycleGAN, self).__init__()
        self.gen_A2B = Generator()  # Generator from domain A to domain B
        self.gen_B2A = Generator()  # Generator from domain B to domain A
        self.disc_A = Discriminator()  # Discriminator for domain A
        self.disc_B = Discriminator()  # Discriminator for domain B
        self.gan_loss = GANLoss()  # Adversarial loss
        self.lambda_cycle = 10.0  # Cycle consistency weight

    def forward(self, real_A, real_B):
        # Generate fake images
        fake_B = self.gen_A2B(real_A)
        fake_A = self.gen_B2A(real_B)
        # Reconstruct images
        cycle_A = self.gen_B2A(fake_B)
        cycle_B = self.gen_A2B(fake_A)

        # Return generated and reconstructed images
        return fake_A, fake_B, cycle_A, cycle_B

    def compute_loss(self, real_A, real_B):
        fake_A, fake_B, cycle_A, cycle_B = self.forward(real_A, real_B)

        # Calculate the adversarial losses
        loss_G_A2B = self.gan_loss(self.disc_B(fake_B), True)
        loss_G_B2A = self.gan_loss(self.disc_A(fake_A), True)

        # Calculate cycle consistency losses
        loss_cycle_A = cycle_consistency_loss(real_A, cycle_A, self.lambda_cycle)
        loss_cycle_B = cycle_consistency_loss(real_B, cycle_B, self.lambda_cycle)

        # Total generator loss
        loss_G = loss_G_A2B + loss_G_B2A + loss_cycle_A + loss_cycle_B

        # Calculate the discriminator losses
        loss_D_A = self.gan_loss(self.disc_A(real_A), True) + self.gan_loss(self.disc_A(fake_A.detach()), False)
        loss_D_B = self.gan_loss(self.disc_B(real_B), True) + self.gan_loss(self.disc_B(fake_B.detach()), False)

        # Total discriminator loss
        loss_D = loss_D_A + loss_D_B

        return loss_G, loss_D
```

#### Part 3: Training and Testing the CycleGAN Model (35 points)

**Task Description**:  
Train your CycleGAN model using one of the  datasets from the provided link. You will implement the training loop, evaluate the model's performance, and visualize the results. This task will give you practical experience with CycleGAN in translating between two distinct image domains.

**Dataset**:  
Select any dataset from the [CycleGAN Datasets](https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/) page. Some popular datasets include:

- **Horse2Zebra**: Translating horse images to zebra images and vice versa.
- **Apple2Orange**: Translating apple images to orange images.
- **Summer2Winter**: Translating summer landscape images to winter landscapes.

**Steps to Prepare the Dataset**:

1. Choose and download your desired dataset from the link above.
2. Unzip the downloaded dataset, which will contain two folders: `trainA` and `trainB`.
3. Implement a custom PyTorch `Dataset` class to load and preprocess the images.

**Task**:

1. **Prepare the Data Loader (5 points)**:
   - Create a custom `Dataset` class to handle image loading and preprocessing.
   - Apply appropriate data transformations such as resizing, random cropping, and normalization.

2. **Implement the Training Loop (20 points)**:
   - Write a training loop that updates the CycleGAN model using the defined loss functions.
   - Save the model weights periodically and log training progress.

3. **Testing and Visualization (10 points)**:
   - Test the CycleGAN model on unseen images.
   - Visualize the results by showing input images, translated images, and reconstructed images.

**Code Skeleton**:

```python
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

# Define the Dataset class
class UnpairedImageDataset(Dataset):
    def __init__(self, root_A, root_B, transform=None):
        """
        Initialize the dataset with paths to domain A and domain B images.

        Args:
            root_A (str): Path to the images in domain A.
            root_B (str): Path to the images in domain B.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.A_images = sorted(os.listdir(root_A))
        self.B_images = sorted(os.listdir(root_B))
        self.root_A = root_A
        self.root_B = root_B
        self.transform = transform

    def __len__(self):
        return max(len(self.A_images), len(self.B_images))

    def __getitem__(self, idx):
        A_img_path = os.path.join(self.root_A, self.A_images[idx % len(self.A_images)])
        B_img_path = os.path.join(self.root_B, self.B_images[idx % len(self.B_images)])

        A_img = Image.open(A_img_path).convert("RGB")
        B_img = Image.open(B_img_path).convert("RGB")

        if self.transform:
            A_img = self.transform(A_img)
            B_img = self.transform(B_img)

        return A_img, B_img

# Define the data transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Initialize the dataset and dataloader
dataset = UnpairedImageDataset(root_A='path/to/trainA', root_B='path/to/trainB', transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

# Initialize the CycleGAN model
cyclegan = CycleGAN()
cyclegan.to('cuda')  # Move model to GPU if available

# Define optimizers
optimizer_G = optim.Adam(list(cyclegan.gen_A2B.parameters()) + list(cyclegan.gen_B2A.parameters()), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(list(cyclegan.disc_A.parameters()) + list(cyclegan.disc_B.parameters()), lr=0.0002, betas=(0.5, 0.999))

# Training Loop
num_epochs = 100
for epoch in range(num_epochs):
    for i, (real_A, real_B) in enumerate(dataloader):
        real_A = real_A.to('cuda')
        real_B = real_B.to('cuda')

        # Zero the parameter gradients
        optimizer_G.zero_grad()
        optimizer_D.zero_grad()

        # Compute the losses
        loss_G, loss_D = cyclegan.compute_loss(real_A, real_B)

        # Backpropagate and optimize
        loss_G.backward()
        optimizer_G.step()
        
        loss_D.backward()
        optimizer_D.step()

        # Print training progress
        if i % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Step [{i}/{len(dataloader)}], Loss_G: {loss_G.item()}, Loss_D: {loss_D.item()}')

    # Save model checkpoints periodically
    torch.save(cyclegan.state_dict(), f'cyclegan_epoch_{epoch}.pth')

# Testing and Visualization
cyclegan.eval()
with torch.no_grad():
    for i, (real_A, real_B) in enumerate(dataloader):
        if i > 5:  # Display results for 5 examples
            break
        real_A = real_A.to('cuda')
        real_B = real_B.to('cuda')
        fake_B, fake_A, _, _ = cyclegan.forward(real_A, real_B)

        # Visualization code to display real and generated images (to be implemented by the student)
```

#### Part 4: Reflection and Summary (10 points)

**Task Description**:  
After completing the implementation and testing of your CycleGAN model, choose two of the following questions to answer. This reflection task aims to help you critically evaluate your work and explore the broader implications of CycleGAN.

**Questions (Choose Two to Answer)**:

1. **Training and Performance Challenges**:
   - Describe a key challenge you faced during the training process (e.g., instability, convergence issues). How did you address this challenge, and what did you learn from it?

2. **Quality of Generated Images**:
   - Evaluate the quality of the generated images. What are some common artifacts or issues you observed? What do you think might be causing these problems?

3. **Content Preservation vs. Style Transfer**:
   - How well does the model balance content preservation and style transfer? Provide an example from your results where the model either succeeded or failed in maintaining the original content.

4. **Potential Improvements**:
   - Suggest one improvement that could enhance the performance of CycleGAN. This could be a change in architecture, loss function adjustment, or a new data augmentation technique.

5. **Applications of CycleGAN**:
   - Based on your experience, suggest a practical application for CycleGAN in a real-world scenario. Explain how CycleGAN could be beneficial in this context.

**Submission Requirements**:
- Write your response to two of the above questions in a markdown cell in your Jupyter Notebook.
- Use clear and concise language. Include visual examples or results if relevant to your discussion.

### Submission Requirements

- Submit your Jupyter Notebook (*.ipynb) with all code cells executed and results visible.
- Include a summary of findings and reflections at the end of the notebook if needed.
- Ensure that your notebook runs without errors when executed from top to bottom.

### Something May Help You

- CycleGAN: https://junyanz.github.io/CycleGAN/
- A Gentle Introduction to CycleGAN for Image Translation: https://machinelearningmastery.com/what-is-cyclegan/
- Code for pytorch-CycleGAN-and-pix2pix: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
- TensorFlow Core CycleGAN Tutorial: https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/cyclegan.ipynb
- PyTorch Colab notebook: https://colab.research.google.com/github/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/CycleGAN.ipynb

