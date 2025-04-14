from tqdm import tqdm
import torch
import torch.nn as nn

def train(model, clip_model, train_loader, optimizer, scheduler, device):
  running_loss = 0.0
  iter_cnt = 0
  correct_sum = 0

  model.to(device)
  model.train()

  total_loss = []
  with tqdm(total=len(train_loader)) as pbar:
      for batch_i, (imgs1, labels) in enumerate(train_loader):
        imgs1 = imgs1.to(device)
        labels = labels.to(device)

        criterion = nn.CrossEntropyLoss(reduction='none')

        output, MC_loss = model(imgs1, clip_model, labels, phase='train')

        loss1 = nn.CrossEntropyLoss()(output, labels)

        loss = loss1 + 5 * MC_loss[1] + 1.5 * MC_loss[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iter_cnt += 1
        _, predicts = torch.max(output, 1)
        correct_num = torch.eq(predicts, labels).sum()
        correct_sum += correct_num
        running_loss += loss

        pbar.update(1)  # Update progress bar for each batch


  scheduler.step()
  running_loss = running_loss / iter_cnt
  acc = correct_sum.float() / float(train_loader.dataset.__len__())
  return acc, running_loss