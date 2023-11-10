# ML_Project_MRI_Reconstruction
MRI Reconstruction 
Data Collection and Preprocessing: We will begin by acquiring the fully sampled MRI data from Kaggle while ensuring data quality and consistency. And choosed an undersampling strategy which will help to reduce the acqusion time significantly and also which is feasible to acquire from the MRI machine.

Undersampling Strategy: To mimic real-world accelerated MRI acquisitions, we will try to design and implement various undersampling strategies. These strategies will allow us to assess our deep learning model's performance under different levels of undersampling, reflecting the diversity of clinical scenarios.

Deep Learning Model Development: We will develop a robust deep learning model, rooted in convolutional neural networks & GAN Architecture (Generative Adversarial Network), to predict the missing k-space data and subsequently reconstruct undersampled MRI images. The model will be designed to learn the intricate relationships between the available k-space data and the image content.

Evaluation and Comparison: The accuracy and effectiveness of our deep learning model will be assessed through rigorous evaluation processes. We will compare the reconstructed images to the original fully sampled MRI data, utilizing established image quality metrics such as Structural Similarity Index (SSIM) and Peak Signal-to-Noise Ratio (PSNR) to quantitatively measure the quality of the reconstruction.

Fine-tuning and Optimization: We will iterate on our deep learning model, incorporating feedback from evaluations, to enhance its performance and efficiency. This iterative process is essential for achieving optimal results and generalizability.

### Workflow:
![image](https://github.com/satyamsangwan1/ML_Project_MRI_Reconstruction/assets/115143488/8a5c2390-f5ad-4cc3-8f77-1b17e5da3bdd)

Ref:
This model architechture was insipired by paper “ReconResNet: Regularised Residual Learning for MR Image Reconstruction of Undersampled Cartesian and Radial Data”
Link of the paper : https://arxiv.org/pdf/2103.09203.pdf
