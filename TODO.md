## Swarm tracker:

For a machine learning classification problem the no of training examples shown to a classifier has to be proportionate to the pattern complexity. A classifier might need more examples to learn that something is a square than to learn something like a uniform background.

Members of a class should be homogeneous mogrphologically, Our initial idea was that the junctions, interiors and exteriors form the classes. But in a collective swarm problem because of the dynamics involved, there can be a large morpological variation inside a class itself. For instance all of the following blue images represent interior pixels, even though there is a large variation in the morphology.

Thus for a network to generalize across all of these, you need to identify the common morphological classes.

Output folder: https://www.dropbox.com/sh/d0hhm878kqq9852/AADrizLzxQzng2NJM3N8gkgZa?dl=0

### Morphological classes which are abundant and thus the network learns them: 

<figure>
    <img src='morphological_classes/pearl.png?raw=true' width="200" height="200" alt='missing' />
    <figcaption>Pearl</figcaption>

</figure>

<figure>
    <img src='morphological_classes/single.png?raw=true' width="200" height="200" alt='missing' />
    <figcaption>Single</figcaption>

</figure>

### Relatively sparse classes on which the network misclassifies: 

<figure>
    <img src='morphological_classes/batman.png?raw=true' width="200" height="200" alt='missing' />
    <figcaption>Batman</figcaption>

</figure>

<figure>
    <img src='morphological_classes/fatneck.png?raw=true' width="200" height="200" alt='missing' />
    <figcaption>Fatneck</figcaption>

</figure>

<figure>
    <img src='morphological_classes/S.png?raw=true' width="200" height="200" alt='missing' />
    <figcaption>S</figcaption>

</figure>

<figure>
    <img src='morphological_classes/U.png?raw=true' width="200" height="200" alt='missing' />
    <figcaption>U</figcaption>

</figure>

Solution: Identify roughly how many morphological classes there are. Weigh down classes by pattern complexity.

### Pipeline:

* Identify common morphological pattern classes.
* Show the network training examples in proportion to the class complexity.
* Evaluate on the video.
* Find the common misclassified morphological patterns. Balance dataset to add more samples of this category. (by cropping them out and generating a new image). Retrain network.
* Repeat till satisfactory convergence.

### Doing
* Identify the misclassified classes and fix the erroneous classes.
* Generate outputs for videos 14 and 15.

### Issues faced:
* Erroneous classifications because of class underrepresentation:
* Memory errors since working with 1080x1080 images: Reduced batch size
* Training time takes too long: Solution: Add multiple gpus and batch run the code on them.
* Setting stuff on google compute engine.
* Larger time for processing multiple videos: Solution: Run on a k80/Make the network sparser(working on it, can still use some improvements)

### Next steps:
* Try to make network sparser to make it faster.
* Running things on multiple gpus to batch process the dataset.
* Use kalman/particle filters to get smoothened trajectories (per frame classifications will be jaggy).
* Train a rnn to get trajectories.

### Things to think about:
* Decomposing the problem into multiple problems: Finding the contours first and then trying to segmennt vs joint end to end.
* Automatically sample the erroneous classes.
* Automated way to identify morphological classes? Will make this useful as a reusable swarm tracking toolkit.
* Contour wise classification.
* Attention mechanisms.

