#%%
import random
import numpy as np
import datetime

#%%
from Deep_learning_homework_2.autograd import Value
from Deep_learning_homework_2.nn import Neuron, Layer, MLP



#%%
model = MLP(3, [4, 4, 1])
print(model)
print("number of parameters", len(model.parameters()))


#%%
xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0] # desired targets


#%%
# loss function
def loss(X, y, model, batch_size=None):
    # inline DataLoader :)
    if batch_size is None:
        Xb, yb = X, y
    else:
        ri = np.random.permutation(X.shape[0])[:batch_size]
        Xb, yb = X[ri], y[ri]
    inputs = [list(map(Value, xrow)) for xrow in Xb]

    # forward the model to get scores
    scores = list(map(model, inputs))

    # svm "max-margin" loss
    losses = [(1 + -yi * scorei).relu() for yi, scorei in zip(yb, scores)]
    data_loss = sum(losses) * (1.0 / len(losses))
    # L2 regularization
    alpha = 1e-4
    reg_loss = alpha * sum((p * p for p in model.parameters()))
    total_loss = data_loss + reg_loss

    # also get accuracy
    accuracy = [((yi).__gt__(0)) == (scorei.data.__gt__(0)) for yi, scorei in zip(yb, scores)]
    return total_loss, sum(accuracy) / len(accuracy)


# optimization
for k in range(20):  #

    # forward
    total_loss, acc = loss(xs, ys, model)

    # backward
    model.zero_grad()
    total_loss.backward()

    # update (sgd)
    learning_rate = 1.0 - 0.9 * k / 100
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    if k % 1 == 0:
        print(f"step {k} loss {total_loss.data}, accuracy {acc * 100}%")
