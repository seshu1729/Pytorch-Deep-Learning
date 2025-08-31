Name: Practical Deep Learning: Master PyTorch in 15 Days\
Source: Udemy\
Link: https://www.udemy.com/course/deep-learning-practical/

```
python -m venv env
env\Scripts\activate.bat

pip install jupyter
jupyter kernelspec list
python -m ipykernel install --user --name=env --display-name "env"
```
==========================

If you are use INTEL GPU:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
```
If you are using Nvidia GPU:
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
Install the required libraries. Uninstall the libraries if you face any issues based on the error logs.
Some more Library packages needed:
```
pip install numpy pandas matplotlib jupyter scikit-learn gradio tqdm huggingface_hub[hf_xet] transformers 
```
=================================




Optimizing Training for Our Neural Network Classifier
If training a neuron with 500,000 iterations takes too long, try reducing the number of iterations to 250,000 and increasing the learning rate, for example, to 0.025.

When working with networks, it is possible to get stuck in local minima for several iterations or more due to the random initialization of weights and biases. If the loss does not decrease for a significant portion of the iterations, rerunning the model might help. This issue depends on factors such as the number of training iterations, the learning rate, activation functions and the optimizer. To resolve it, experiment with these parameters until a working solution is found.

