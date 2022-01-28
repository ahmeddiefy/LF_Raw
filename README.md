# LF_Raw
Light Field Reconstruction Using Residual Networks on Raw Images
We have proposed a learning-based LF reconstruction method. To effectively explore the non-local property of 4D LF, we adopted the raw LF representation which enabled the network to understand and model the relationship well and thus restore more texture de-tails and provide better quality. We initialized the views to be reconstructed using the nearest view method, along with the raw LF representation, the task was transformed from image reconstruction into image-to-image translation. Our method improves the av-erage PSNR over the second-best method by 0.64 dB.

For testing
Download test data and place them at ./data/TrainingData/Test/{Dataset Folder}
Run test_and_save.py to get the results mentioned
