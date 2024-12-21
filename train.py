def train(model, train_loader, criterion, optimizer, device):
    model.train()
    for batch_idx, (data, target, length) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
       
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f'Training... [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                    f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
