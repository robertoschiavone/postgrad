from graph import draw_dot  # noqa
from core.mlp import MLP

if __name__ == "__main__":
    net = MLP(3, [4, 4, 1])

    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    ys = [1.0, -1.0, -1.0, 1.0]

    for k in range(101):
        # forward pass
        ypred = [net(x) for x in xs]
        loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

        # backward pass
        for p in net.parameters():
            p.grad = 0.0
        loss.backward()

        # update
        for p in net.parameters():
            p.data += -0.1 * p.grad

        if k % 10 == 0 or k < 10:
            print(f"{k}: {loss.data:.8f}")

    print(ypred)
