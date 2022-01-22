# Pyt-Ex
just something for easy use of PyTorch


# Where we are

It's been a while since I started this project with the idea of building something which would be helpful in construction and training of ANN's for simple classification tasks. It didn't take me long to discover that constructiong NN's dynamically and training them to converge is not as easy as it sounds. Especially due to ton's of parameters, few of them being
    1) How deep the neural net is (currently the code can make 1 and 2 hidden layers).
    2) `BatchNorm` and `Dropout` layers (Especially while you're trying to compare with and without these layers).
    3) Limits in using various kind of Loss functions, as the training loop for `MSE` is different from `CrossEntropyLoss`.
    4) Techniques such as skip-connections are also hard to employee with current codebase.
    5) Moreover The biggest of all is Hyper-parameter tuning. This is biggest bottleneck at the moment as i'm conflicted in writting `GridSearchCv` from scratch or change my codebase to complement existing libraries which does this stuff.

And the list goes on.

# This might as well be a lost casue

Yes, unless I change the direction of where this is heading might as-well pack everything up and start picking up new things. My initial goal to build this for researchers from non computer science background to make use of Pytorch. But, many seems to be already liking tools such as `MatLab`, so idk.

# possible future direction
  yet to do.