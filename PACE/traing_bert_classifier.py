import torch,yaml,os
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
from datasets import load_dataset
from torch.optim import AdamW
from transformers import AutoTokenizer,BertModel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from util import setup_logger
from torchmetrics import Accuracy
from bert_model import *
torch.cuda.empty_cache()

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
os.makedirs("bert_model_weight",exist_ok=True)
logger=setup_logger("log/trian_snli_classifier.log")

def trainer(model,dataloader,optimizer,scheduler,criterion):
    t_loss=0.0
    evl_old_acc=0
    epochs=10
    logger.info(f"Total Epoch {epochs}, Start to train !")
    for epoch in range(pre_epoch,epochs):
        model.train()
        for idx,(batch,labels) in enumerate(p:=tqdm(dataloader.train_loader)):

            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            t_loss+=loss.item()
            acc=accuracy(outputs,labels)
            p.set_postfix_str(f"Epoch {epoch+1}/{epochs}: Loss {loss.item():.5f}, lr: {optimizer.param_groups[0]['lr']}, acc :{acc:.2f}")
            # if acc>tri_old_acc and not idx%500 and idx >=500:
            # # Save the trained model
            #     torch.save({
            #             'model_state_dict': model.state_dict(),
            #         }, f"bert_model_weight/bert_snli_classifier_train_M3_{acc:.2f}.pth")
            #     tri_old_acc=acc
            #     logger.info(f"Write : bert_model_weight/bert_snli_classifier_train_M3_{acc:.2f}.pth")
        t_mean_loss=t_loss/len(dataloader.train_loader)
        scheduler.step(t_mean_loss)
        logger.info(f"Epoch {epoch+1}/{epochs}, Train Pass, Start to eval")
        evl_old_acc=validation(model,dataloader.valid_loader,criterion,evl_old_acc)

def validation(model,valid_loader,criterion,evl_old_acc):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch,labels in (p:=tqdm(valid_loader)):
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)

            val_loss+=loss.item()
            acc=accuracy(outputs,labels)

    if acc>=evl_old_acc:
        # Save the trained model
        torch.save({
                'model_state_dict': model.state_dict(),
            }, f"bert_model_weight/bert_snli_classifier_eval_M3_{acc:.2f}.pth")
        evl_old_acc=acc
        logger.info(f"Write : Validation Loss: {val_loss}, Validation Accuracy: {acc}")
    else:
        logger.info(f"Validation Loss: {val_loss}, Validation Accuracy: {acc}")
    return evl_old_acc


if __name__=="__main__":

    # Load pre-trained BERT model for sequence classification
    # model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    # Instantiate the BERT-based classifier
    model = BERTSequenceClassifier(num_classes=3)
    snli_dataloader=snli_dataloader()
    prestraned_weight_path=f""
    pre_epoch=0

    if os.path.isfile(prestraned_weight_path):
        pre_data=torch.load(prestraned_weight_path)
        model.load_state_dict(pre_data['model_state_dict'])
        logger.info(f"Load From {prestraned_weight_path}")
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler=ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=0.00001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    accuracy = Accuracy(task="multiclass", num_classes=3).to(device)
    model.to(device)
    trainer(model,snli_dataloader,optimizer,scheduler,criterion)

