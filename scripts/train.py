from configs.config import args
from torch.utils.data import DataLoader 
import torch
from tqdm import tqdm 
from typing import Tuple
from utils.EvalMetrics import measure_model_performance

def train_and_eval_model(model,
                         train_loader: DataLoader,
                         val_loader: DataLoader,
                         loss_fn: torch.nn.Module,
                         number_of_epochs: int = args.epochs, 
                         ) -> Tuple[list, list, list, list]:

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate) # weight_decay=1e-3
    model.to(args.device)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_recall , val_recalls = [] , []

    for epoch in range(number_of_epochs):

        # -------- TRAINING -------- #

        model.train()
        running_loss, running_acc, running_recall = 0 , 0 , 0 
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{number_of_epochs} - Training"):
            x, y = x.to(args.device), y.to(args.device)
            y_pred = model(x)
            y_pred = y_pred.squeeze()
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            acc, recall = measure_model_performance(y_pred, y)
            running_recall += recall.item()
            running_acc += acc.item()
        epoch_train_recall = running_recall/len(train_loader)
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = running_acc / len(train_loader)
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        train_recall.append(epoch_train_recall)

        # -------- EVALUATION -------- #

        model.eval()
        val_loss, val_acc , val_recall=0 ,0  ,0 
        with torch.inference_mode():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{number_of_epochs} - Validation"):
                x, y = x.to(args.device), y.to(args.device)
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

    return train_losses, train_accuracies , train_recall, val_losses, val_accuracies , val_recalls


                  

