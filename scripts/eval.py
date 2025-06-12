from configs.config import args
from torch.utils.data import DataLoader 
import torch
import tqdm 
from typing import Tuple
from utils.EvalMetrics import measure_model_performance

def eval_model(model,
                         test_loader: DataLoader,
                         loss_fn: torch.nn.Module,
                        #  pack_path_way:Optional[Callable[[int , int], torch.tensor]]
                         ) -> Tuple[list, list, list, list]:
        
        val_losses, val_accuracies , val_recalls = [] , [] 
        model.eval()
        val_loss, val_acc , val_recall=0 ,0  ,0 
        with torch.inference_mode():
            for x, y in tqdm(test_loader):
                x, y = x.to(args.device), y.to(args.device)
                y_pred = model(x)
                y_pred = y_pred.squeeze()
                loss = loss_fn(y_pred, y)

                val_loss += loss.item()
                acc, recall = measure_model_performance(y_pred, y)
                val_recall += recall.item()
                val_acc += acc.item()

        epoch_val_loss = val_loss / len(test_loader)
        epoch_val_acc = val_acc / len(test_loader)
        epoch_val_recall = val_recall/len(test_loader)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        val_recalls.append(epoch_val_recall)



        return  val_losses, val_accuracies , val_recalls


                  

