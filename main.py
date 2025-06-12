from scripts.train import train_and_eval_model
from models.conv2dplus1d import FireDetectorConv2D1D
from utils.Dataset import data_loader
from utils.EvalMetrics import loss_fn
from configs.config import args

loader = data_loader()

result = train_and_eval_model(FireDetectorConv2D1D ,loader['sample_train_loader'], loader['sample_test_loader'], loss_fn , args.epochs )

print(result)
