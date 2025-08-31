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

