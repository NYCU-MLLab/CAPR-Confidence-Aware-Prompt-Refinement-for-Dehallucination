import torch,yaml,os
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
from datasets import load_dataset
from torch.optim import AdamW
from transformers import AutoTokenizer,BertModel
from util import setup_logger
from torchmetrics import Accuracy

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Add a classification layer on top of BERT
class BERTSequenceClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BERTSequenceClassifier, self).__init__()
        self.bert_model_name=config['trian_bert_config']['model_name']
        self.bert = BertModel.from_pretrained(self.bert_model_name)
        self.bert.train()
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.softmax=nn.Softmax(dim=-1)
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.softmax(self.fc(pooled_output))
        return logits


class snli_dataloader:
    def __init__(self) -> None:
        # Load SNLI dataset
        snli_dataset = load_dataset('snli')
        filtered_dataset = snli_dataset.filter(lambda x: x["label"] in [0, 1,2])
        self.bert_model_name=config['trian_bert_config']['model_name']

        # Load BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
        self.tokenizer.model_max_length=1024

        # Prepare DataLoader
        self.train_loader = DataLoader(filtered_dataset["train"], batch_size=128, shuffle=True,collate_fn=self.collect_fn)
        self.valid_loader = DataLoader(filtered_dataset["validation"], batch_size=128,collate_fn=self.collect_fn)

    def collect_fn(self,batch):
    # 获取批次中每个样本的input_ids、attention_mask和labels
        text = [example['premise']+self.tokenizer.sep_token+example['hypothesis'] for example in batch]
        token_result=self.tokenizer(text,truncation=True,padding=True,max_length=1024,return_tensors='pt')
        label_change=[]
        for example in batch:
            label_change.append(example['label'])
            # if example['label']==2:
            #     label_change.append(1)
            # elif example['label']==2:
            #     label_change.append(0)
        labels = torch.tensor(label_change)
        return token_result, labels
