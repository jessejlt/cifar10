# Explorations in Machine Learning

Just a simple annotated example of a convolutional neural network for classifying the [CIFAR 10](https://www.kaggle.com/c/cifar-10/data) dataset.

# Install

Suggest using [virtualenv](https://virtualenv.pypa.io/en/stable/) with python3. Install your dependencies `pip install -r requirements.txt`.

# Execution

* Run model `python dcnn.py`
* Visualize `tensorboard --logdir=./logs` and visit `127.0.0.1:6006`
* Warning! If you're doing this on a laptop, plug it in first!

If when running tensorboard you see the error 
`AttributeError: module 'site' has no attribute 'getsitepackages'

Get your site packages via:

```
~/Code/ML/cifar10 master*
(cifar10) ❯ python
Python 3.5.2 (default, Sep 14 2016, 11:28:32) 
[GCC 6.2.1 20160901 (Red Hat 6.2.1-1)] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from distutils.sysconfig import get_python_lib
>>> print(get_python_lib())
/home/jesse/Code/ML/cifar10/lib/python3.5/site-packages
>>> 
```

and modify `bin/tensorboard` with your site packages

```
# for mod in site.getsitepackages():
    for mod in [ '/home/jesse/Code/ML/cifar10/lib/python3.5/site-packages' ]:
```
