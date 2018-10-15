Stacked Autoencoder Neuroevolution

# Paper
*saene.pdf* is a shortened version of the master thesis as a research paper.

# Stacked Autoencoder Neuroevolution (SAENE)
This project implements a neuroevolution strategy for autoencoders. It
implements a stacked autoencoder model using tensorflow.

## Requirements
To run the code python3.6 is required. All package requirements are listed in
the *requirements.txt*

## Getting Started
I recommend to run this code in a virtual python environment. To do this,
install *virtualenv* using pip with `pip3 install virtualenv` on Linux.

To create the virtual environment use *virtualenv* on Linux or *venv* on
MacOs:

```bash
virtualenv -p python3 machine-learning
python3 -m venv machine-learning
```

Activate the virtual environment by running `source
machine-learning/bin/activate`

Next install the requirements from the *requirements.txt*
`pip install -r requirements.txt`

If the requirements change the *requirements.txt* can be updated using
`pip freeze > requirements.txt`

# docs
Sphinx project that contains the documentation for the code.

Update the documentation by running `make html` from the *docs/* folder. The
virtual environment has to be activated for this to work.

To regenerate the package documenation (after major changes) delete the
contents of the *source/* directory and run `sphinx-apidoc -o source/ ../saene/`
from the *docs/* directory.

