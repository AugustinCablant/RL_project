# Why do we need Experience Replay?
- Neural networks take a batch of data ( If we trained it with single samples, each sample and the corresponding gradients would have too much variance, and the network weights would never converge.)
- Why doing it by experience replay ? Because sequential actions are highly correlated with one another and are not randomly shuffled, as the network would prefer.

# Why do we need a second neural network (the target network) ?
- By employing a second network that doesn’t get trained, we ensure that the Target Q values remain stable, at least for a short period. 