import torch
import torch.optim as optim

# Inicijalizacija parametara
# Inicijalizacija na standardnu normalnu razdiobu (1 x 1) + oznaka da se za tu varijablu računa gradijent
a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# Definiranje skupa za učenje
X = torch.tensor([1, 2])
Y = torch.tensor([3, 5])

# Optimizacijski postupak - stohastički gradijentni spust -> optimiramo varijable a i b
optimizer = optim.SGD([a, b], lr=0.1)

for i in range(100):
    # Definicija afinog regresijskog modela
    Y_ = a * X + b

    # Računanje kvadratnog gubitka
    diff = (Y - Y_)
    loss = torch.sum(diff ** 2)

    # Računanje gradijenata
    loss.backward()

    # Optimiziranje parametara
    optimizer.step()

    # Postavljanje gradijenata na 0 (jer bi se u protivnom akumulirao gradijente)
    optimizer.zero_grad()

    print(f'step: {i}, loss:{loss}, Y_:{Y_}, a:{a}, b {b}')
