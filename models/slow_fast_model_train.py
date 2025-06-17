import torch
from tqdm import tqdm 
from utils.Dataset import DataLoader
from typing import Tuple
from utils.EvalMetrics import measure_model_performance
from configs.config import args
from torch import nn


slow_fast_model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50' , pretrained=True)
in_features = slow_fast_model.blocks[-1].proj.in_features
slow_fast_model.blocks[-1].proj = nn.Linear(in_features , 1)


def slow_fast_model_train(model, train_loader:DataLoader , val_loader:DataLoader, 
                          loss_fn,
                          number_of_epochs:int ,
                          pathway_alpha:int)-> Tuple[list , list , list , list]:

    def pack_pathway_output(frames, alpha):
        fast_pathway = frames
        slow_pathway = frames[:, :, ::alpha, :, :]  # temporal subsampling
        return [slow_pathway, fast_pathway]
    
    def padd_to_32(data:torch.tensor) : 
            T = data.shape[2]
            if T == 32 : 
                return data 
            elif T < 32: 
                padd = 32 - T 
                st1 =  data[:,:,-1:,:,:].repeat(1, 1, padd , 1, 1)
                stacked = torch.cat([st1, data] , dim=2)
                return stacked
            else: 
                return data[:, :, :32, : , :]
            

    def transform_data(data:torch.tensor): 

        import albumentations as A

        transform = A.Compose([
            A.SmallestMaxSize(max_size=256),  # Equivalent to ShortSideScale(256)
            A.CenterCrop(height=224, width=224),
            A.Normalize(mean=[0.45]*3, std=[0.225]*3),
        ])

        # input shape N, C , D , H , W 
        p_data = data.permute(0, 2 , 1, 3 ,4)
        N, D, C, H, W = p_data.shape
        r_data = p_data.reshape(N * D , H, W , C)
        n_data = r_data.cpu().numpy()
        augmented = transform(images=n_data)
        augmented_video_frames = augmented['images']

        data = augmented_video_frames.reshape(N, D , C , H , W)   
        data = data.transpose(0 , 2 , 1, 3, 4)  
        data = torch.from_numpy(data).to(args.device)
        data = pack_pathway_output(data , pathway_alpha)   # this transofrms frame to slow and fast double frames 
        
        return data
        
        
    optimizer = torch.optim.Adam(params=model.parameters(), lr = args.learning_rate)
    model.to(args.device)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_recalls , val_recalls = [] , []

    for epoch in range(number_of_epochs):
        running_loss , running_acc , running_recall = 0 , 0 , 0 
        for x , y in tqdm(train_loader , desc=f'Epoch {epoch+1}/{number_of_epochs} - Training' ): 
            # ----- TRAINING ----- #
            x , y = x.to(args.device) , y.to(args.device)
            x = padd_to_32(x)
            x = transform_data(x)

            model.train() 
            ypred = model(x).squeeze()
            loss = loss_fn(ypred, y)
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()
            acc , recall = measure_model_performance(ypred, y)
            running_loss += loss.item() 
            running_acc  += acc 
            running_recall += recall
        epoch_train_recall = running_recall/len(train_loader)
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = running_acc / len(train_loader)
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        train_recalls.append(epoch_train_recall)        

    # ----- EVALUATION ---- # 

        model.eval()
        val_loss, val_acc , val_recall= 0.0, 0.0 , 0.0
        with torch.inference_mode():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{number_of_epochs} - Validation"):
                x, y = x.to(args.device), y.to(args.device)
                x = padd_to_32(x)
                x = transform_data(x)
                y_pred = model(x)
                y_pred = y_pred.squeeze()
                loss = loss_fn(y_pred, y)
                val_loss += loss.item()
                acc, recall = measure_model_performance(y_pred, y)
                val_recall += recall.item()
                val_acc += acc.item()

        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = val_acc / len(val_loader)
        epoch_val_recall = val_recall/len(val_loader)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        val_recalls.append(epoch_val_recall)

        print(f"[Epoch {epoch+1}] Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

    return train_losses, train_accuracies , train_recalls, val_losses, val_accuracies , val_recalls






