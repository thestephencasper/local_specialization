# Detecting Modularity in Deep Neural Networks

Shlomi Hod*, Stephen Casper*, Daniel Filan*, Cody Wild, Andrew Critch, Stuart Russell

A link to the paper will be provided as soon as it is available on arXiv. Check again in a couple of days. 

## Research Environment Setup

### Docker

Useful: [Lifecycle of Docker Container](https://medium.com/@nagarwal/lifecycle-of-docker-container-d2da9f85959)

#### Building the image (done **ONCE** per machine)

Clone the repository and change to the `devops` directory.

```bash
docker build -t humancompatibleai/nn-clustering .
```

#### Creating a container (done **ONCE** by one user per one machine)

In other words, if you run this commands on, for example, `svm` - you don't need to do so again. You're supposed then to have a *container* there already.

First, you need a port number to your Jupyter Netbook - pick up a random number (with your favoriate generator) in the range 8000-8500.
We pick up a random number so you don't collide with existing notebooks on that machine.

First run: 

- **Remove the comments before**, and 
- **Replace** `<PORT NUMBER>` with your random port number (also in the instructions that will come later)

```bash
docker run \
-it \
-p <PORT NUMBER>:8888 \
--rm \
--name nn_clustering-$(whoami) \
--runtime=nvidia \  # REMOVE, if you don't have GPU
--mount type=bind,source=/scratch,target=/scratch \  # REMOVE, if not on perceptron or svm machines
humancompatibleai/nn-clustering:latest \
bash
```

And then type

```bash
bash build.sh
```

NB: to leave the container, use ctrl-P ctrl-Q. Typing `exit` will destroy the container.

#### Running the container

```bash
docker exec \
-it nn_clustering-$(whoami) \
bash
```
### Coding Environment

Requirements: Python 3.7 (not tested with earlier versions)

The environment is set up in a Python virtual environment. To do so:

1. Clone this repository

2. Install `graphviz`
   1. Ubuntu/Debian: `apt intall graphviz`
   2. MacOS: `brew install graphviz`

3. Install with `pipenv install --dev`

4. On MacOS **only**, you will need to install `pygraphviz` separatly:
   `pipenv run pip install pygraphviz --install-option="--include-path=/usr/local/Cellar/`

5. To set up the dependencies and finish, type `cd nn_clustering` and `./build.sh`

6. Install packages with `pipenv install --system`. Additionally, post-hoc install: `pipenv install image-classifiers==1.0.0 tensorflow-datasets`

## Instructions

This requires the Imagenet2012 validation dataset. You will need to register at [http://www.image-net.org/download-images](http://www.image-net.org/download-images) and obtain a link to download the dataset. Then execute the following.
```bash
mkdir datasets
mkdir datasets/imagenet2012
cd datasets/imagenet2012
wget [YOUR LINK HERE]
```
That's all! When running experiments with imagenet models, tfrecords will be created automatically from the `.tar` file.

See `shells/prepare_all.sh` for commands to make datasets, train networks, cluster, and perform experiments. We use `make` with a `Makefile` to streamline running things. But rather than simply running `shells/prepare_all.sh`, you may wish to only parts of it. It may take a long time.
