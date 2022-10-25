import torch
from tqdm import tqdm
import gc
import time
import numpy as np
from collections import defaultdict
import torch
from torch import nn
from torchmetrics import MetricCollection, Accuracy, F1Score
import os

def run_training(config, model, train_dataloader, valid_dataloader, optimizer, lr_scheduler):
    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))

    device = torch.device(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    metrics_fn = MetricCollection([Accuracy(num_classes=config.DATA.NUM_CLASS).to(device), F1Score(num_classes=config.DATA.NUM_CLASS).to(device)])
    model.to(device)

    
    start = time.time()
    best_acc        = -np.inf
    best_f1         = -np.inf
    best_epoch      = -1
    history = defaultdict(list)
    num_epochs = config.TRAIN.EPOCHS
    for epoch in range(1, num_epochs + 1): 

        gc.collect()
        print(f'Epoch {epoch}/{num_epochs}', end='\n')
        train_loss = train_step(config, epoch, model, train_dataloader, optimizer, lr_scheduler, criterion, metrics_fn, device)

        val_loss, val_scores = valid_step(config, model, valid_dataloader, criterion, metrics_fn, device)
        val_acc, val_f1 = val_scores['acc'], val_scores['f1']
    
        history['Train Loss'].append(train_loss)
        history['Valid Loss'].append(val_loss)
        history['Valid Accuracy'].append(val_acc)
        history['Valid F1Score'].append(val_f1)
        
        # Log the metrics
        print(f'Valid Accuraacy: {val_acc:0.2f} | Valid F1 Score: {val_f1:0.3f}\n')
        if best_acc < val_acc:
            best_acc = val_acc
            save_state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'best_acc': best_acc,
                'epoch': epoch,
                'config': config
            }
            save_path = f'{config.OUTPUT}/ckpt_epoch_{epoch}.pth'
            print(f"{save_path} saving.......")
            torch.save(save_state, save_path)
            print(f"{save_path} saved!")

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60)
    )
    
    return history


def train_step(config, epoch, model, dataloader, optimizer, lr_scheduler, criterion, metrics_fn, device):
    model.train()
    num_steps = len(dataloader)
    dataset_size = 0
    running_loss = 0.0
    metrics = {
        'acc': 0.0,
        'f1': 0.0
    }

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train ')
    
    for step, data in pbar:
        _, imgs, input_ids, attn_masks, labels = data

        imgs = imgs.to(device)
        input_ids = input_ids.to(device)
        attn_masks = attn_masks.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        if lr_scheduler is not None:
            lr_scheduler.step_update((epoch*num_steps + step))
        #logits, attn_map
        logits, _ = model(imgs, input_ids, attn_masks)
        loss = criterion(logits, labels)
        loss.backward()

        # TODO: Clip grad
        optimizer.step()

        running_loss += (loss.item() * config.DATA.BATCH_SIZE)
        dataset_size += config.DATA.BATCH_SIZE

        metric_scores = metrics_fn(logits.softmax(-1), labels)
        metrics['acc'] += metric_scores['Accuracy'].cpu().detach().numpy()
        metrics['f1'] += metric_scores['F1Score'].cpu().detach().numpy()
        
        epoch_loss = running_loss / dataset_size
        acc_score = metrics['acc'] / (step+1)
        f1_score = metrics['f1'] / (step+1)

        mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        pbar.set_postfix(
            F1_score=f"{f1_score:0.3f}",
            Accuracy=f"{acc_score:0.2f}",
            gpu_mem=f"{mem:0.2f} GB",
            train_loss=f"{epoch_loss:0.4f}"
        )
        
    torch.cuda.empty_cache()
    gc.collect()

    return epoch_loss


@torch.no_grad()
def valid_step(config, model, dataloader, criterion, metrics_fn, device):

    model.eval()
    dataset_size = 0
    running_loss = 0.0
    metrics = {
        'acc': 0.0,
        'f1': 0.0
    }
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Valid ')

    for step, data in pbar:
        _, imgs, input_ids, attn_masks, labels = data

        imgs = imgs.to(device)
        input_ids = input_ids.to(device)
        attn_masks = attn_masks.to(device)
        labels = labels.to(device)

        logits, _ = model(imgs, input_ids, attn_masks)
        loss = criterion(logits, labels)

        running_loss += (loss.item() * config.DATA.BATCH_SIZE)
        dataset_size += config.DATA.BATCH_SIZE
        epoch_loss = running_loss / dataset_size

        metric_scores = metrics_fn(logits.softmax(-1), labels)
        metrics['acc'] += metric_scores['Accuracy'].cpu().detach().numpy()
        metrics['f1'] += metric_scores['F1Score'].cpu().detach().numpy()
        
        acc_score = metrics['acc'] / (step+1)
        f1_score = metrics['f1'] / (step+1)

        mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        pbar.set_postfix(
                         F1_score=f"{f1_score:0.3f}",
                         Accuracy=f"{acc_score:0.2f}",
                         gpu_mem=f"{mem:0.2f} GB",
                         valid_loss=f"{epoch_loss:0.4f}",
                        )

    metrics['acc'] = acc_score
    metrics['f1'] = f1_score
    torch.cuda.empty_cache()
    gc.collect()

    return epoch_loss, metrics

def predict_with_test(config, model, dataloader, le):
    
    import pandas as pd

    device = torch.device(config.DEVICE)
    submit_df = pd.read_csv(f'{config.DATA.PATH}/sample_submission.csv')

    with torch.no_grad():
        for data in tqdm(dataloader):
            img_ids, imgs, input_ids, attn_masks = data
            
            imgs = imgs.to(device)
            input_ids = input_ids.to(device)
            attn_masks = attn_masks.to(device)

            logits, _ = model(imgs, input_ids, attn_masks)
            logits = logits.softmax(dim=-1).detach().cpu().numpy()

            preds = np.argmax(logits, axis=-1)
            labels = le.inverse_transform(preds)
            for img_id, label in zip(img_ids, labels):
                submit_df.loc[submit_df['id'] == img_id, 'cat3'] = label
    submit_df.to_csv(f'{config.DATA.PATH}/sample_submission.csv')
