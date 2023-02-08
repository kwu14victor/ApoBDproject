import os
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
#from torchsummary import summary
import tqdm
import numpy as np
#from vit_pytorch import ViT
#from vit_pytorch.vit_for_small_dataset import ViT as ViTsmall
from pytorch_pretrained_vit import ViT

class ImageClassificationBase(nn.Module):
    
    def training_step(self, batch):
        images, labels = batch 
        images, labels = images.cuda(), labels.cuda()
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        acc = accuracy(out, labels) 
        
        return loss, acc
    
    def validation_step(self, batch):
        images, labels = batch 
        images, labels = images.cuda(), labels.cuda()
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels).cuda()   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        mat = matrices(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc, 'mat': mat}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses, torch.stack will combine tensors along a new dimension
        batch_accs = [x['val_acc'] for x in outputs]    
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        batch_mat = [x['mat'] for x in outputs]
        epoch_mat = np.sum(np.array(batch_mat), axis=0)#torch.stack(batch_mat).mean()   
        TP,TN,FP,FN = epoch_mat[0], epoch_mat[1], epoch_mat[2], epoch_mat[3]
        P, R, F1 = TP/(TP+FP), TP/(TP+FN), 2*TP/(2*TP+FP+FN)   
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item(), 'Precision':P, 'Recall':R, 'F1':F1}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1score: {:.4f}".format(
            epoch, result['train_loss'], result['train_acc'], result['val_loss'], result['val_acc'], result['Precision'],result['Recall'],result['F1']))

class MyAlexNet(ImageClassificationBase):
    def __init__(self, args):
        super().__init__()
        if args.mode=='train':
            model = models.alexnet(pretrained=True)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
            model.act2 = nn.Softmax()
            #for _,v in model.features.named_parameters():
            #    v.requires_grad=False
            #for n,v in model.classifier.named_parameters():
            #     if int((n.split('.')[0]))!=6 and int((n.split('.')[0]))!=4:
            #       v.requires_grad=False
            self.network = model 
        #elif args.mode=='extract':
        else:
            model = models.alexnet(pretrained=True)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
            model.act2 = nn.Softmax()
            #for _,v in model.features.named_parameters():
            #    v.requires_grad=False
            #for n,v in model.classifier.named_parameters():
            #     if int((n.split('.')[0]))!=6 and int((n.split('.')[0]))!=4:
            #       v.requires_grad=False
            self.network = model 
            '''
            alex = models.alexnet(pretrained=True)
            fea = nn.Sequential(* list(alex.features.children())[:-1]) 
            #alex.features
            for _,v in fea.named_parameters():
                v.requires_grad=False
            self.network = fea
            '''
        
    def forward(self, xb):
        return self.network(xb)

class MyResnet(ImageClassificationBase):
    def __init__(self, args):
        super().__init__()
        if args.mode=='train':
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, 2)
            model.act2 = nn.Softmax()

            self.network = model 
        else:
            model = models.resnet18(pretrained=True)
            #model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
            model.fc = nn.Linear(model.fc.in_features, 2)
            model.act2 = nn.Softmax()
            self.network = model 
        
    def forward(self, xb):
        return self.network(xb)

class MyResnet50(ImageClassificationBase):
    def __init__(self, args):
        super().__init__()
        
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.act2 = nn.Softmax()
        self.network = model 
        
    def forward(self, xb):
        return self.network(xb)

'''
class MyResnet50(ImageClassificationBase):
    def __init__(self, args):
        super().__init__()
        
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.act2 = nn.Softmax()
        self.network = model 
        
    def forward(self, xb):
        return self.network(xb)
'''
class MyInception(ImageClassificationBase):
    def __init__(self, args):
        super().__init__()
        model = models.googlenet(aux_logits=False,pretrained=True)#pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.act2 = nn.Softmax()
        self.network = model 
    def forward(self, xb):
        return self.network(xb)

class MyVIT(ImageClassificationBase):
    
    def __init__(self, args):
        super().__init__()
        if args.mode=='train':
            #model = ViT(image_size = 51, patch_size = 17, num_classes = 2, dim = 256, depth = 3, heads = 4, mlp_dim = 512, dropout = 0.2, emb_dropout = 0.2)
            model = ViT(image_size = args.size, patch_size = args.psize, num_classes = 2, dim = 128, depth = 2, heads = 4, mlp_dim = 256, dropout = 0.2, emb_dropout = 0.2)
            model.act2 = nn.Softmax()
            #for _,v in model.features.named_parameters():
            #    v.requires_grad=False
            #for n,v in model.classifier.named_parameters():
            #     if int((n.split('.')[0]))!=6 and int((n.split('.')[0]))!=4:
            #       v.requires_grad=False
            self.network = model 
        #elif args.mode=='extract':
        #    alex = models.alexnet(pretrained=True)
        #    fea = nn.Sequential(* list(alex.features.children())[:-1]) 
            #alex.features
        #    for _,v in fea.named_parameters():
        #        v.requires_grad=False
        #    self.network = fea
        
    def forward(self, xb):
        return self.network(xb)


class MyVIT2(ImageClassificationBase):

    def __init__(self, args):
        super().__init__()
        if args.mode=='train':
            #model = ViTsmall(image_size = 51, patch_size = 17, num_classes = 2, dim = 256, depth = 3, heads = 4, mlp_dim = 512, dropout = 0.2, emb_dropout = 0.2)
            model = ViTsmall(image_size = 51, patch_size = 17, num_classes = 2, dim = 128, depth = 2, heads = 4, mlp_dim = 256, dropout = 0.2, emb_dropout = 0.2)
            model.act2 = nn.Softmax()
            #for _,v in model.features.named_parameters():
            #    v.requires_grad=False
            #for n,v in model.classifier.named_parameters():
            #     if int((n.split('.')[0]))!=6 and int((n.split('.')[0]))!=4:
            #       v.requires_grad=False
            self.network = model 
        #elif args.mode=='extract':
        #    alex = models.alexnet(pretrained=True)
        #    fea = nn.Sequential(* list(alex.features.children())[:-1]) 
            #alex.features
        #    for _,v in fea.named_parameters():
        #        v.requires_grad=False
        #    self.network = fea
        
    def forward(self, xb):
        return self.network(xb)

class MyVIT3(ImageClassificationBase):

    def __init__(self, args):
        super().__init__()
        if args.mode=='train':
            model = ViT('B_16_imagenet1k', pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, 2)
            self.network = model 
        
    def forward(self, xb):
        return self.network(xb)

class MyVITB16(ImageClassificationBase):

    def __init__(self, args):
        super().__init__()
        if args.mode=='train':
            model = ViT('B_16', pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, 2)
            self.network = model 
        
    def forward(self, xb):
        return self.network(xb)

class MyVITB32(ImageClassificationBase):

    def __init__(self, args):
        super().__init__()
        if args.mode=='train':
            model = ViT('B_32', pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, 2)
            self.network = model 
        
    def forward(self, xb):
        return self.network(xb)

class CNNLSTM(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNLSTM, self).__init__()
        self.CNN = models.alexnet(pretrained=True)
        self.resnet.fc = nn.Sequential(nn.Linear(self.CNN.fc.in_features, 300))
        self.lstm = nn.LSTM(input_size=300, hidden_size=256, num_layers=3)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
       
    def forward(self, x_3d):
        hidden = None
        for t in range(x_3d.size(1)):
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :, :])  
            out, hidden = self.lstm(x.unsqueeze(0), hidden)         

        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        x = self.fc2(x)
        return x


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
    
def matrices(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    
    T, F = (preds == labels).cpu().numpy(), (preds != labels).cpu().numpy()
    P, N = (labels==0).cpu().numpy(), (labels==1).cpu().numpy()
    TP = sum(np.logical_and(T,P))
    TN = sum(np.logical_and(T,N))
    FP = sum(np.logical_and(F,N))
    FN = sum(np.logical_and(F,P))
    
    #print(T)
    #TP, TN, FP, FN = (torch.sum(preds==labels and labels==0).item())/len(preds), (torch.sum(preds==labels and labels==1).item())/len(preds), (torch.sum(preds!=labels and labels==1).item())/len(preds), (torch.sum(preds!=labels and labels==0).item())/len(preds)
    #TP, TN, FP, FN = (torch.sum(T and P).item())/len(preds), (torch.sum(T and N).item())/len(preds), (torch.sum(F and N).item())/len(preds), (torch.sum(F and P).item())/len(preds)
    #print(TP, TN, FP, FN)
    #P, R, F1 = TP/(TP+FP), TP/(TP+FN), 2*TP/(2*TP+FP+FN)
    
    return (np.array([TP,TN,FP,FN]))#torch.tensor([TP,TN,FP,FN,P,R,F1])    