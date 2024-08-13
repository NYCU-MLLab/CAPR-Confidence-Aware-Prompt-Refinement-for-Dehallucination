from datasets import load_dataset
from torch.utils.data import DataLoader
from datetime import datetime
from util import *
import os,re

class qadataset_dataloader:
    def __init__(self,dataset_path="din0s/asqa",split='train',batch_size=1,shuffle=True):
        super(qadataset_dataloader,self).__init__()
        self.dataset_path=dataset_path
        self.split=split
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.get_loader()

    def setup_collect_fn(self): ## "gsm8k",DateUnd,Prf-Law,Biz-Ethics
        if self.dataset_path=="din0s/asqa":
            dataset_name="din0s/asqa"
            self.collect_fn=self.asqa_collate_fn
            self.dataset = load_dataset(dataset_name)

        elif self.dataset_path=="natural_questions": # Short Ans ; EM Score
            self.collect_fn=self.naturalqa_collect_fn
            self.dataset=self.Load_natural_qa(20)

        elif self.dataset_path=="ChilleD/StrategyQA": # True/ False
            dataset_name="ChilleD/StrategyQA"
            self.collect_fn=self.stretagy_qa_collect_fn
            self.dataset = load_dataset(self.dataset_path)

        elif self.dataset_path=="gsm8k":
            dataset_name="gsm8k"
            self.collect_fn=self.gsm8k_collect_fn
            self.dataset = load_dataset(self.dataset_path,"main")

        elif self.dataset_path=="triviaQA":
            dataset_name="lucadiliello/triviaqa"
            self.collect_fn=self.triviaqa_collect_fn
            self.dataset = load_dataset(dataset_name)

    def get_loader(self):

        self.setup_collect_fn()
        if self.split in self.dataset:
            self.trainloader=DataLoader(self.dataset[self.split],batch_size=self.batch_size,shuffle=self.shuffle,collate_fn=self.collect_fn,num_workers=1,drop_last=True)
        else:
            self.trainloader = DataLoader(self.dataset, batch_size=self.batch_size, collate_fn=self.collect_fn,shuffle=self.shuffle,num_workers=1,drop_last=True)

    def Load_natural_qa(self,idx):
        qa_dict=[]
        for i in range(idx):
            datapath=f"natural_qa_dataet/natual_qa_{i}.json"
            old_data=load_checkpoint(datapath)
            qa_dict+=old_data
        return qa_dict

    def asqa_collate_fn(self,batch): ## long_ans

        isurl=True
        res=[[i['ambiguous_question'],i['wikipages'][0]['url'],isurl] for i in batch]
        long_ans=[i['annotations'][0]['long_answer'] for i in batch]
        return res,long_ans

    def naturalqa_collect_fn(self,batch): ## Long ans or short ans
        res=[[i['Question'],i['Document'],i['isurl']] for i in batch]
        short_ans=["".join(i['Short_answer']) for i in batch]
        long_ans=["".join(i['long_answer'])for i in batch]
        return res,short_ans

    def stretagy_qa_collect_fn(self,batch): ## True / False
        res=[]
        isurl=False
        res=[[i['question'],i['facts'],isurl]for i in batch]
        answer=[i['answer'] for i in batch]
        return res,answer

    def gsm8k_collect_fn(self,batch):

        res=[[i["question"],"",False] for i in batch]
        ans=[i['answer']for i in batch]
        ground_Truth=[re.findall(r'\d+',i['answer']).pop()for i in batch]

        return res,ans

    def triviaqa_collect_fn(self,batch):
        res=[[i['question'],i['context'].replace("[PAR]","").replace("[DOC]","").replace("[TLE]",""),False] for i in batch]
        ans=[sorted(i['answers'],key=lambda x:len(x),reverse=False)[-1] for i in batch]

        return res,ans

class eval_dataloader:
    def __init__(self,dataset_path,batch_size,purpose='compare',tokenizer="",shuffle=False) -> None:
        self.tokenizer = tokenizer
        self.shuffle=shuffle
        self.dataset = load_checkpoint(dataset_path)
        if purpose=="compare":
            self.loader=DataLoader(list(self.dataset),batch_size=len(self.dataset),collate_fn=self.parallel_simi_collate_fn,shuffle=self.shuffle,drop_last=True)
        elif purpose=="acc":
            self.loader=DataLoader(list(self.dataset),batch_size=len(self.dataset),collate_fn=self.acc_collate_fn,shuffle=self.shuffle,drop_last=True)

    def acc_collate_fn(self,batch):
        # print(batch)
        ans=[i['Answer'] for i in batch]
        ground_Truth=[i['Ground_Truth']for i in batch]
        res=[[i['Document'],i['Doc_Ans_simi'],i['Confidence'],i['Prompt']]for i in batch]
        return res,ans,ground_Truth

    def parallel_simi_collate_fn(self,batch):
        ans=[i['Answer'] for i in batch]
        long_ans=[i['Ground_Truth']for i in batch]
        res=[[i['Question'],i['Document'],i['Confidence'],i['Explanation'],i['Prompt']] for i in batch]
        return res,ans,long_ans


if __name__=="__main__":
    qa_loader=qadataset_dataloader("triviaQA",split='train',batch_size=1,shuffle=True).trainloader
    for batch,gournd_truth in qa_loader:
        print(batch)
        print(gournd_truth)
        break

    # simi_loader=eval_dataloader("response_result/ChilleD_StrategyQA_gpt-3.5-turbo-0125_vanilla_QA_2024_05_12.json",1,'compare').loader
    # for res,ans,long_ans in simi_loader:
    #     print(ans)

