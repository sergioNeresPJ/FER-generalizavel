from .train import train
from .test import test
import torch

def train_model(model, clip_model, train_loader, test_loader, optimizer, scheduler, device, best_model_path, epochs=100, patience = 5):
    best_acc = 0
    no_improvement = 0
    
    for i in range(1, epochs + 1):
        train_acc, train_loss = train(model, clip_model, train_loader, optimizer, scheduler, device)
        test_acc, test_loss = test(model, clip_model, test_loader, device)
        print('epoch: ', i, 'acc_test: ', test_acc, 'acc_train: ', train_acc)
    
        # Early stopping logic with patience
        if test_acc > best_acc:
            best_acc = test_acc
            no_improvement = 0  # Reset patience counter on improvement
            torch.save({'model_state_dict': model.state_dict(),}, best_model_path)
        else:
            continue
            no_improvement += 1  # Increment patience counter on no improvement
    
        if no_improvement == patience:
            print(f"Early stopping after {i} epochs with no improvement in test accuracy")
            break  # Exit the training loop if patience is exhausted
            
        with open('results.txt', 'a') as f:
            f.write(str(i)+'_'+str(test_acc)+'\n')