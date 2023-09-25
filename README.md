# peatrain-v2023
2023 rewrite of the 'Peatear' neural network code, in both TensorFlow and PyTorch versions.

This code is based on the 'Peatear' neural network code developed by Lachlan Mitchell in 2018, for detecting Faba bean sprouts at early emergence. The code has been streamlined for use with the Master of Data Science Intership at the Biometry Hub in the Winter Trimester of 2023.

- `peatrain_*.py` trains the neural network (with the same model layer structure as developed by Lachlan Mitchell) on image files listed in `annotations.csv`. Both TensorFlow and PyTorch versions are available, as a showcase of the similarities and differences between the two toolsets.
- `peapredict_*.py` is a demonstration program, that takes a given image and runs it through the compiled neural network to show how the CNN model makes its predictions about whether a given image is a plant or not. 
