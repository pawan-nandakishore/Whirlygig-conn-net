## Swarm tracker:

For a machine learning classification problem the no of training examples shown to a classifier has to be proportionate to the pattern complexity. For example a pattern like the pixels with intensity == 0px is background. A classifier might need more examples to learn that something is a square than to learn something like a uniform background.

Issue is in a swarm problem, the 
Problems: Concept of a class. Members of a class should be homogeneous mogrphologically, Our initial idea was that the junctions, interiors and exteriors form the classes. But in a collective swarm problem because of the dynamics involved, there can be a large morpological variation inside a class itself. For instance all of these morphological_classes represent nieghbourhoods of junctions. [morphological_classes]

Thus for a network to generalize across all of these, you need to identify the common morphological classes.

Morphological classes which are abundant and thus the network learns them instantly: single whirligig (elliptical) ![](morphological_classes/single.png?raw=true  =100x100), pearl,![](morphological_classes/single.png?raw=true =100x100) H shaped pattern(two necks).

Relatively sparse classes on which the network misperforms: batman pattern,![](morphological_classes/single.png?raw=true) snake pattern,![](morphological_classes/single.png?raw=true) u pattern (only one neck),![](morphological_classes/single.png?raw=true) lightsaber pattern, fatneck pattern![](morphological_classes/single.png?raw=true)

Solution: Identify roughly how many morphological classes there are. Weigh down classes by pattern complexity.

### Pipeline:

* Identify common morphological pattern classes.
* Show the network training examples in proportion to the class complexity.
* Evaluate on the video.
* Find the common misclassified morphological patterns. Balance dataset to add more samples of this category. (by cropping them out and generating a new image). Retrain network.
* Repeat till satisfactory convergence.

### Doing
* Identify the misclassified classes and fix the erroneous classes.

### Issues faced:
* Erroneous classifications because of class underrepresentation:
* Memory errors since working with 1080x1080 images.
* Training time takes too long: Solution: Add multiple gpus and batch run the code on them.
* Setting stuff on google compute engine.
* Solution: Run on a k80/Make the network sparser(working on it, can still use some improvements)

### Next steps:
* Try to make network sparser to make it faster.
* Running things on multiple gpus to batch process the dataset.
* Use kalman/particle filters to get smoothened trajectories (per frame classifications will be jaggy).
* Train a rnn to get trajectories.

### Things to think about:
* Decomposing the problem into multiple problems: Finding the contours first and then trying to segmennt vs joint end to end.
* Automated way to identify morphological classes? Possibly some form Will make this useful as a reusable swarm tracking toolkit.


