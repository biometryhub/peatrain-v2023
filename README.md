# peatrain-v2023
2023 rewrite of the 'Peatear' neural network code, in both TensorFlow and PyTorch versions.

This code is based on the 'Peatear' neural network code developed by Lachlan Mitchell in 2018, for detecting Faba bean sprouts at early emergence. The code has been streamlined for use with the Master of Data Science Intership at the Biometry Hub in the Winter Trimester of 2023.

- `peatrain_*.py` trains the neural network (with the same model layer structure as developed by Lachlan Mitchell) on image files listed in `annotations.csv`. Both TensorFlow and PyTorch versions are available, as a showcase of the similarities and differences between the two toolsets.
- `peapredict_*.py` is a demonstration program, that takes a given image and runs it through the compiled neural network to show how the CNN model makes its predictions about whether a given image is a plant or not. 

## GPU container

Requires Docker with the NVIDIA Container Toolkit.

Build:

```sh
docker build -t peatrain-gpu .devcontainer
```


Run:

```sh
# Check GPU is found
docker run --rm --gpus=all peatrain-gpu python -c "import torch, tensorflow as tf; print('torch', torch.cuda.is_available()); print('tf', tf.config.list_physical_devices('GPU'))"

# Train, then predict on a sample image
docker run --rm --gpus=all -v "$PWD":/workspace -w /workspace peatrain-gpu python peatrain_pytorch.py
docker run --rm --gpus=all -v "$PWD":/workspace -w /workspace peatrain-gpu python peapredict_pytorch.py nnmodel.pth "images/testimage02-0-1005,8156-1069,8220.jpg"
```

For TensorFlow, use `peatrain_tensorflow.py` and `peapredict_tensorflow.py nnmodel.keras ...` instead of the
`pytorch` counterparts.
