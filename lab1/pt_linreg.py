import torch
from torch.optim import SGD


def mean_square_loss(Y_, Y):
    """
    uvedeno zbog uprosječivanja gubitka za svaku točku
    """
    return torch.mean((Y_ - Y) ** 2)


def perform_linear_regression(X, Y_, n_iter: int, learning_rate: float):
    a = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)

    optimizer = SGD([a, b], lr=learning_rate)
    for i in range(n_iter):
        Y = a * X + b
        loss = mean_square_loss(Y_, Y)
        loss.backward()
        optimizer.step()

        if i % 10 == 0 or i == n_iter - 1:
            print(f"Iteration {i}:")
            print(f"\tloss = {loss.data}")
            print(f"\ta = {a.data[0]}")
            print(f"\tb = {b.data[0]}")
            print(f"\tGrads:")

            diff = Y - Y_
            my_a_grad = 2 * torch.mean(diff * X)
            my_b_grad = 2 * torch.mean(diff)

            print(f"\t\tmy_a_grad = {my_a_grad.data}")
            print(f"\t\tmy_b_grad = {my_b_grad.data}")

            print(f"\t\ta_grad = {a.grad[0]}")
            print(f"\t\tb_grad = {b.grad[0]}")
            print()

        optimizer.zero_grad()


if __name__ == '__main__':
    X = torch.tensor([1, 2])
    Y_ = torch.tensor([3, 5])
    perform_linear_regression(X, Y_, 1000, 0.1)
