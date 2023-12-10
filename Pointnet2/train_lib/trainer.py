from tqdm import tqdm
import importlib
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from data_utils.data_map import labels_poss as labels
from train_lib.eval_utils import *
import warnings
warnings.filterwarnings("ignore")

def downstream_iter_callback(params, model, writer, scheduler, batch_loss, batch_acc, batch_ious, train_epoch, is_train):
    # after each epoch we log the losses on tensorboard
        if is_train:
            writer.add_scalar(
                'training/learning_rate',
                scheduler.optimizer.state_dict()['param_groups'][0]['lr'],
                train_epoch,
            )
        # loss
        writer.add_scalar(
            'training/loss' if is_train else 'validation/loss',
            batch_loss,
            train_epoch,
        )
        # accuracy
        writer.add_scalar(
            'training/acc' if is_train else 'validation/acc',
            batch_acc,
            train_epoch,
        )
        # mean iou
        writer.add_scalar(
            'training/miou' if is_train else 'validation/miou',
            batch_ious[-1],
            train_epoch,
        )
        # per class iou
        for class_num in range(0,batch_ious.shape[0]-1):
            writer.add_scalar(
                f'training/per_class_iou/{labels[class_num]}' if is_train else f'validation/per_class_iou/{labels[class_num]}',
                batch_ious[class_num].item(),
                train_epoch,
            )
        if is_train:
            torch.save(model.state_dict(), f'{params.model_checkpoint}/checkpoint_{train_epoch}.pt')
        
def validation(params, writer, scheduler, model, criterion, val_loader, epoch):
    model.eval()
    losses = []
    accuracies = []
    ious = []
    with torch.no_grad():
        with tqdm(total=len(val_loader), desc="Validation", unit="batch") as v:
            for i , batch in enumerate(val_loader):
                if len(batch.points) > 1:
                    if params.use_gpu:
                        batch.to(torch.device("cuda:0"))                
                    outputs = model(torch.stack(batch.points).float())
                    loss = criterion(outputs, batch.labels.squeeze().long())
                    losses.append(loss.cpu().item())
                    accuracies.append(accuracy(outputs, batch.labels.squeeze().long()))
                    ious.append(intersection_over_union(outputs, batch.labels.squeeze().long()))
                    v.set_postfix(accuracy=np.nanmean(np.array(ious), axis=0)[-1], loss=np.mean(losses))
                    v.update()
        downstream_iter_callback(params, model, writer, scheduler, np.mean(losses),
                                 np.nanmean(np.array(accuracies), axis=0)[-1],
                                 np.nanmean(np.array(ious), axis=0), epoch, is_train=False)


def training(params, training_loader, validation_loader, loss):

    _model = importlib.import_module(params.model_name)
    model = _model.PointNet2SemSegMSG(params).cuda()
    writer = SummaryWriter(f'runs')
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    criterion = loss 
    model.train()
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2)
    # Train
    for epoch in range(params.epochs):
        # metrics
        losses = []
        accuracies = []
        ious = []
        # iterate over dataset
        with tqdm(total=len(training_loader), desc="Training-Epoch " + str(epoch), unit="batch") as t:
            for i , batch in enumerate(training_loader):
                if len(batch.points) > 1:
                    if params.use_gpu:
                        batch.to(torch.device("cuda:0"))   
                    optimizer.zero_grad()
                    outputs = model(torch.stack(batch.points).float())
                    loss = criterion(outputs, batch.labels.squeeze().long())
        
                    losses.append(loss.cpu().item())
                    loss.backward()
                    optimizer.step()        
                    accuracies.append(accuracy(outputs, batch.labels.squeeze().long()))
                    ious.append(intersection_over_union(outputs, batch.labels.squeeze().long()))
                    t.set_postfix(accuracy=np.nanmean(np.array(ious), axis=0)[-1], loss=np.mean(losses))
                    t.update()
            #saving the results
            downstream_iter_callback(params, model, writer, scheduler, np.mean(losses),
                                     np.nanmean(np.array(accuracies), axis=0)[-1],
                                     np.nanmean(np.array(ious), axis=0), epoch, is_train=True)
            #validation
            validation(params, writer, scheduler, model, criterion, validation_loader, epoch)
            # Update learning rate
            if epoch in params.lr_decays:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= params.lr_decays[epoch]
    
            scheduler.step(np.mean(losses))      
            torch.cuda.empty_cache()
            torch.cuda.synchronize(torch.device("cuda:0"))




