
![horizontal line]

<a name="_leajue2ys1lr"></a>Machine Learning 

<a name="_81g3rsq4ohcy"></a>**Undersampled MRI Reconstruction**

![horizontal line](img/Aspose.Words.fc62e1f8-a784-40b8-898d-8aecdb19005c.002.png)
# <a name="_arolcxe0i15c"></a>**Model 1** 
### <a name="_zge4vrg40ckk"></a>**Input: Under-sampled MRI image (256\*256\*1)**
### <a name="_ixngeju7ryom"></a>**Output:  Reconstructed MRI image (256\*256\*1)**
### <a name="_cvdpdacuumjt"></a>**Dataset: Kaggle Igg-mri-segmentation (Fully sampled dataset)**
### <a name="_d3x7he32wi21"></a>**Loss function: SSIM**
### <a name="_dytmbuw7e4t"></a>**Epochs: 100 with batch size 32**
### <a name="_tt2ahebwi1f0"></a>**GitHub link: <https://github.com/satyamsangwan1/ML_Project_MRI_Reconstruction/tree/main/FINAL_SUBMISSION/First%20Model>**
**Paper link**: <https://arxiv.org/pdf/2103.09203.pdf>
### <a name="_k7oplackpoex"></a>**Introduction:**
The input image for the model is **256\*256\*1.**		 	 	 			
We applied Fourier transforms on MRI images to produce kspaces.
For constructing the undersampled images, we applied the undersampling mask to the kspaces to produce undersampled kspaces.
Then inverse Fourier transform to these undersampled kspaces to undersampled
images.
### <a name="_m8iimuc11ozk"></a>**Model architecture** 
Paper Link: https://arxiv.org/pdf/2103.09203.pdf

The model architecture we used in our model is inspired by the **ReconResNet: Regularised Residual Learning for MR Image Reconstruction of Undersampled Cartesian and Radial Data** paper, for which the link is provided above.

- Input Downsampling: The network begins with two downsampling blocks that reduce the input size by half.
- ` `Residual Blocks: It consists of 34 residual blocks. Each block outputs the same size as its input, and the input is added to its output before being forwarded to the next residual block.
- Output Upsampling: Following the residual blocks, the network has two upsampling blocks, double the input size, to restore the original image size.

### ![](img/Aspose.Words.fc62e1f8-a784-40b8-898d-8aecdb19005c.003.png)
### ![](img/Aspose.Words.fc62e1f8-a784-40b8-898d-8aecdb19005c.004.png)
###
###
### <a name="_241bnaklrt4b"></a><a name="_bkhdd0pp3h0d"></a><a name="_dd8gb9ndyezi"></a><a name="_ukvet9sxvjt9"></a><a name="_ajt9gouytbaa"></a>	**Model Results:**

Avg SSIM between target and blurred image = 0.7734475678476569

Avg SSIM between target and deblurred image = 0.839262833766049

Avg PSNR between target and blurred image = 26.997193178883474

Avg PSNR between target and deblurred image = 29.871343676221052



![](img/Aspose.Words.fc62e1f8-a784-40b8-898d-8aecdb19005c.005.png)

![](img/Aspose.Words.fc62e1f8-a784-40b8-898d-8aecdb19005c.006.png)

**We can see that the PSNR value of the image increased from 26.997 to 29.871, and the SSIM also increased from 0.773 to 0.839**



# <a name="_wesfri7x5xxv"></a>**Model 2 ( Conditional-GAN )**
### <a name="_7njz1kz4kpf2"></a>**Input: Under-sampled kspace  (256\*256\*2)**
### <a name="_smf8eomw9vhb"></a>**Output:  Reconstructed MRI image (256\*256\*1)**
### <a name="_19k20mp94x03"></a>**Dataset: Kaggle Igg-mri-segmentation (Fully sampled dataset)**
### <a name="_uvtfzpb7kab5"></a>**Loss function: SSIM & Wasserstein loss**
### <a name="_vlcb3tz3h16t"></a>**Epochs: 50 with batch size 32**
### <a name="_yas12a3bohtf"></a>**GitHub link:**
<https://github.com/satyamsangwan1/ML_Project_MRI_Reconstruction/tree/main/FINAL_SUBMISSION/Second%20Model>

**Paper\_link:<https://www.researchgate.net/publication/332672833_A_Brief_Review_on_MRI_Images_Reconstruction_using_GAN>**
### <a name="_p1c9rmunz5k7"></a>**Introduction:**
The input image for the model is **256\*256\*1.**		 	 	 			
We applied Fourier transforms on MRI images to produce kspaces.
For constructing the undersampled images, we applied the undersampling mask to the kspaces to produce undersampled kspaces.
Then inverse Fourier transform to these undersampled kspaces to undersampled
images.
### <a name="_coisaxcmar3e"></a>**Model architecture** 
Paper Link:<https://www.researchgate.net/publication/332672833_A_Brief_Review_on_MRI_Images_Reconstruction_using_GAN>

- The generator architecture employs a hierarchical structure with down-sampling and up-sampling layers, incorporating batch normalisation, leaky rectified linear units (ReLU), and residual connections to enhance feature extraction and information flow.
- The final layer of the generator produces a single-channel output using hyperbolic tangent activation, translating input k-space data into synthetic MRI images, with the overall design focused on optimising the generation of high-quality and detailed MRI reconstructions.

![](img/Aspose.Words.fc62e1f8-a784-40b8-898d-8aecdb19005c.007.png)


![](img/Aspose.Words.fc62e1f8-a784-40b8-898d-8aecdb19005c.008.png)
### <a name="_z52nlodhvjlv"></a>**Results:** 
The outcomes of the implemented model were suboptimal, potentially attributable to insufficient training epochs and architectural considerations. The limited number of training iterations may have adversely affected the model's performance, underscoring the importance of extending the training duration to enhance convergence and overall efficacy. 

Additionally, the architectural specifications, particularly the transformation of k-space input into the corresponding MRI images, may necessitate reevaluation and refinement to address potential limitations and optimise the model's predictive capabilities.
# <a name="_77mbjchbw51v"></a>**Model 3 ( Deblur-GAN )**
### <a name="_fwvo89672whl"></a>**Input: Under-sampled MRI image (256\*256\*1)**
### <a name="_j3k58jrauz0a"></a>**Output:  Reconstructed MRI image (256\*256\*1)**
### <a name="_wj9m9f4zoo2e"></a>**Dataset: Kaggle Igg-mri-segmentation (Fully sampled dataset)**
### <a name="_8fgl9dcvnlp4"></a>**Loss function: Wasserstein loss & Perceptual loss**
### <a name="_p48jotlsnzhi"></a>**Epochs: 50 with batch size 32**
### <a name="_wpfga3nk5164"></a>**Evaluation metrics: PSNR & SSIM**
### <a name="_7qik8j8ztewx"></a>**GitHub link:**
<https://github.com/satyamsangwan1/ML_Project_MRI_Reconstruction/tree/main/FINAL_SUBMISSION/Third%20Model>

**Paper link**: <https://arxiv.org/pdf/1711.07064.pdf>

### <a name="_13hr8obe6rc7"></a>**Introduction:**
The input image for the model is **256\*256\*1.**		 	 	 			
In undersampled MRI data, we observed that the MRI images get blurred. A deblurring technique, specifically the Deblur-GAN (Generative Adversarial Network), becomes imperative to remove the blur effect. The Deblur-GAN can enhance the clarity and fidelity of the undersampled MRI images, ensuring improved diagnostic precision and accuracy in medical imaging applications.
### <a name="_rqnj9hk0cqfl"></a>**Model architecture** 
Paper Link:<https://arxiv.org/pdf/1711.07064.pdf>

The model architecture we used in our model is inspired by the **DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks** paper, for which the link is provided above.

![](img/Aspose.Words.fc62e1f8-a784-40b8-898d-8aecdb19005c.009.png)

- Two strided convolution blocks (stride: 1/2).
- Nine residual blocks (ResBlocks) each comprising convolution, instance normalisation, and ReLU activation.
- Dropout regularisation (probability: 0.5) after the initial convolution layer in each ResBlock.

![](img/Aspose.Words.fc62e1f8-a784-40b8-898d-8aecdb19005c.010.png)

![](img/Aspose.Words.fc62e1f8-a784-40b8-898d-8aecdb19005c.011.png)
### <a name="_jgko6slxxie5"></a>	**Model Results:**
### <a name="_oghno94gcuqy"></a>**Evaluation metrics: PSNR & SSIM**

Avg SSIM between target and blurred image = 0.0233629009358551

Avg SSIM between target and deblurred image = 0.03885910028705605

Avg PSNR between target and blurred image = 0.9760757083722632

Avg PSNR between target and deblurred image = 1.1516601466059513


**We can see that the PSNR value of the image increased from 0.976 to 1.151, and the SSIM also increased from 0.023 to 0.038**



![](img/Aspose.Words.fc62e1f8-a784-40b8-898d-8aecdb19005c.012.png)![](img/Aspose.Words.fc62e1f8-a784-40b8-898d-8aecdb19005c.013.png)

![](img/Aspose.Words.fc62e1f8-a784-40b8-898d-8aecdb19005c.014.png)




### <a name="_ikxfpo1wlku2"></a>**Comparison of the Models:**

### <a name="_wcnpmgr4k5yq"></a>**In the first model, the PSNR value of the image is increased by 1.1 times the original value and the SSIM by 1.08 times the original.**
### <a name="_z3iyq7saokcg"></a>**In the third model, the PSNR value of the image is increased by 1.17 times the original value and the SSIM  by 1.65 times the original.**

### <a name="_1y0cgdcuwpv1"></a>**Hence, the best final model is Model 3, which uses DeblurGAN.**
### <a name="_o7wjp8sthju7"></a>**Still, the model that we have made is giving not-so-good images due to a smaller number of epochs due to computational constraints.**
### <a name="_lq4n4t1yl2g"></a>**The model can be improved by doing more number of epochs.**



![horizontal line]


[horizontal line]: img/Aspose.Words.fc62e1f8-a784-40b8-898d-8aecdb19005c.001.png









# ML_Project_MRI_Reconstruction (Phase 1)
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
