
## S4

#### Fully Connected Layers:
> Fully connected layers are NO LONGER used until and unless required, large number of parameters, not good for converting 2D information to 1D as that leads to loss of valuable spatial information including loss of translational, rotational, skew invariance in comparison to operations or Convolutions on 2D data.


#### Modern Architectures - Post-2014:
> Modern architectures removed using FC and instead develop Fully Convolutional Networks.

>     ResNet is the latest among the above. You can clearly note 4 major blocks.
>      The total number of kernels increases from 64 > 128 > 256 > 512 as we proceed from the first block to the last (unlike what we discussed where at each  block we expand to 512. Both architectures are correct, but 64  ... 512 would lead in lesser computation and parameters.


#### SoftMax:
> Scales the inputs to be the sum=1 but is NOT PROBABILITY, it's more likelihood in terms of interpretation.The softmax function is often used in the final layer of a neural network-based classifier, one advantage of using the softmax at the output layer is that it improves the interpretability of the neural network.. Now we use `log_softmax` as it is Negative Log of Likelihood which is better than normal `softmax` as it scales the correct class to lower loss value thus allowing network to force fine tuning of weights towards making correct class scores higher. `log(softmax)` is slower and mathematically unstable hence in-built `log_softmax` should be used. To stabalize, the max of the entire vector is subtracted from the vector and then softamx is carried out to avoid over or underflow of values.


#### MaxPooling:
>    used when we need to reduce channel dimensions
>    not used close to each other
>    used far off from the final layer
>    used when a block has finished its work (edges-gradients/textures-patterns/parts-of-objects/objects
>    nn.MaxPool2D()

#### Batch Normalization:
>Normalizes the incoming batch at every layer of the network to make amplitudes more prominent by scaling the values of the channel so that next immediate layer can figure out features with more clarity and confidence in order to make better decision by combination of features. It's an essential ingredient of modern DNN architectures and allows having deeper layers and higher learning rates for better training of networks.
>    used after every layer
>    never used before last layer
>    indirectly you have sort of already used it!
>    nn.BatchNorm2d()

#### DropOut:
>    A regularization technique that randomly drops weights during training to avoid over-dependence on specific features and helps network to focus on other more relevant features for better decision making.
>    used after every layer
>    used with small values
>    is a kind of regularization
>    not used at a specific location
>    nn.Dropout2d()
