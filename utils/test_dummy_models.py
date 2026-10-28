import torch



def reshaped_params(H, L):
    range_net = torch.nn.Sequential(
        torch.nn.Linear(2, H),
        *[torch.nn.Linear(H, H) for _ in range(L)],
        torch.nn.Linear(H, 1),
    )

    first, hidden, last = range_net[0], range_net[1:-1], range_net[-1]
    parts = [first.weight.reshape(-1), first.bias]
    if hidden:
        parts += [torch.stack([l.weight for l in hidden]).reshape(-1),
                    torch.stack([l.bias for l in hidden]).reshape(-1)]
    parts += [last.weight.reshape(-1), last.bias]
    for part in parts:
        print(part.shape)
    parts = torch.cat(parts)
    print(parts.shape)

    
if __name__ == '__main__':
    reshaped_params(16, 4)
