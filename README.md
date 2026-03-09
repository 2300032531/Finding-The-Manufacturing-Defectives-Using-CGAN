# Finding-The-Manufacturing-Defectives-Using-CGAN

**1️⃣ Project Overview**
This project implements a Conditional Generative Adversarial Network (cGAN) to generate synthetic industrial surface defect images across multiple defect categories.
The system includes:
Data preprocessing pipeline
Conditional Generator network
Discriminator network
Custom training loop
Image monitoring module
CNN-based defect classifier for evaluation
Streamlit-based deployment (optional local use)

**🎯 Project Goals**
Generate realistic multi-class defect images
Improve dataset balance for rare defect types
Support industrial surface inspection research
Enable interactive synthetic defect generation

**🔹 Module-Wise Implementation**
**Module 1 — Data Pipeline & Preprocessing**
Dataset
Industrial surface defect datasets such as:
NEU Surface Defect Dataset
MVTec Anomaly Detection Dataset
Custom metal surface defect dataset
Preprocessing Steps
Load images from dataset folder
Resize images to 64×64 or 128×128
Convert images to RGB
Normalize pixel values from [0,255] → [-1,1]
Encode defect labels (one-hot encoding for cGAN)
Create batches using TensorFlow / PyTorch DataLoader
Purpose
Prepare clean, label-aware, standardized images for stable conditional GAN training.

**Module 2 — Conditional Generator Network**
The Generator creates defect images based on noise + class label.
Architecture
Input:
Random noise vector (z)
Defect class label
Concatenate noise and label
Dense layer + reshape
ConvTranspose layers (upsampling)
Batch Normalization
ReLU activation
Tanh output layer
Output
Synthetic defect image for selected class.

<img width="638" height="511" alt="image" src="https://github.com/user-attachments/assets/bc8b6788-b678-403f-a790-baafee52cca4" />


**Module 3 — Discriminator Network**
The Discriminator evaluates real vs fake images conditioned on labels.
Architecture
Input:
Image (real or generated)
Corresponding defect label
Convolutional layers
LeakyReLU activation (0.2)
Dense layer
Sigmoid output
Loss Function
Binary Cross Entropy (BCE)
Optimizer
Adam
Learning rate = 0.0002
Beta1 = 0.5

**Module 4 — Custom Training Loop**
Training Process
Sample random noise
Generate fake images
Train Discriminator on:
Real images
Fake images
Train Generator to fool Discriminator
Repeat for 100–300 epochs
Monitoring
Track Generator loss (G_loss)
Track Discriminator loss (D_loss)
Save sample images periodically
Save model checkpoints

**Module 5 — Image Monitoring & Evaluation**
Visual Monitoring
Generated image grids
Loss curves
Per-class image samples
Evaluation
CNN-based classifier to evaluate realism
Diversity check across defect categories
Mode collapse detection

**Module 6 — Deployment (Colab / Optional Streamlit)**
In Google Colab:
Generate synthetic defect images directly in notebook
Visualize outputs inline
Download generated images
Optional (Local):
Run Streamlit interface for interactive generation

**🛠️ Technologies Used**
Python
TensorFlow / Keras or PyTorch
NumPy
Matplotlib
OpenCV
Google Colab

**▶️ How to Run**
1️⃣ Open the project notebook in Google Colab
2️⃣ Upload or mount the dataset (if required)
3️⃣ Install required libraries (if not already installed)
4️⃣ Run all cells sequentially
5️⃣ Monitor training progress:
Generator loss (G_loss)
Discriminator loss (D_loss)
Generated defect image samples
6️⃣ After training completes:
View synthetic defect images
Save generated samples
Evaluate using CNN classifier (if included)

**📌 Conclusion**
This project demonstrates the implementation of a Conditional GAN (cGAN) for multi-class industrial surface defect image generation. The model improves dataset balance and supports industrial inspection AI systems while being fully executable within Google Colab.
