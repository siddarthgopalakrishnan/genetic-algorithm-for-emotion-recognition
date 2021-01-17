# Genetic-Algorithm-for-feature-subset-selection
Implementation of Genetic Algorithm for finding the best subset of features for emotion detection from an image using the CK+ dataset and Openface toolkit.

## Dataset extraction
- This model was trained on The Extended Cohn-Kanade Dataset (CK+) which is a complete [dataset](https://drive.google.com/file/d/1MGWKN8_95Y_yk0qJ92Humyu_5ZsxI8v0/view?usp=sharing) for action unit and emotion-specified expression
- The images from the dataset were then processed using the [OpenFace toolkit](https://cmusatyalab.github.io/openface/) to obtain the final dataset
- Manually included labels to the dataset in order to train the model for the respective emotion

## Features implemented
- Random initialization of population chromosomes
- Fitness for the individuals (feature subset) was defined using the mean accuracy for classifying emotions
- Selection methods:
    1. Roulette-wheel selection
    2. Rank-based selection
    3. Tournament selection
- Crossover methods:
    1. k-point crossover
    2. Uniform crossover
- Mutation methods:
    1. Bit-swap mutation
    2. Bit-flip mutation
- Population update methods:
    1. Generational update
    2. Weak-parents update

## Models used for training
- Implemented SVM and Logistic Regression from the [sklearn](https://scikit-learn.org/stable/) Python library
- Implemented Neural Network using [Keras](https://www.tensorflow.org/api_docs/python/tf/keras)
- SVM was found to give the best accuracy with the fastest convergence
