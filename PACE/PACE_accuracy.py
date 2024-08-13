import torch
from prompt_strategy import prompter
from qadataset_dataloader import qadataset_dataloader,eval_dataloader
import multiprocessing as mp
from util import *
from evaluate_confidence_score import evaluate_score
import json,os
from tqdm import tqdm
import os,json
from datetime import datetime
from infer_snli_cls import snli_similarity
import yaml
logger = setup_logger(f'log/response_{datetime.now().strftime("%Y%m%d")}.log')
os.environ["TOKENIZERS_PARALLELISM"] = "True"

class Accuracy:
    def __init__(self,dataset_path,api_model,task,key,stretagy,data_prompt_amount,eval_batch_size,acc_model,simi_path):
        ## generate setting
        self.dataset_path=dataset_path
        self.re_gen_times=5
        self.data_prompt_amount=data_prompt_amount
        self.eval_batch_size=eval_batch_size
        self.task=task
        ## init
        self.acc_model=acc_model
        self.Stretagy=stretagy
        self.prompter=prompter()
        self.api_model=api_model
        self.api_key=key
        self.simi_path=simi_path

        self.eval_loader=eval_dataloader(self.simi_path,self.eval_batch_size,'acc').loader

        self.acc_datapath=self.simi_path.replace(f'.json',f'_{self.acc_model}.json')

        logger.info(f"ACC datapath: {self.acc_datapath}")

        if not os.path.isfile(self.simi_path):
            logger.error(f"{self.simi_path} Do not Exists")
            exit()

    def acc_worker(self,acc_list,batch,acc,ans,gound_truth):
        acc_list.append({
            'Prompt':batch[3],
            'Answer':ans,
            'Ground_Truth':gound_truth,
            'Accuracy':float(acc),
            'Confidence':batch[2],
            'Doc_Ans_simi':batch[1],
            'Document':batch[0],
        })
    def acc_main(self,):
        '''
        Get Document and Response Similarity Score for Confidence Calibration
        Option : ["gpt-3.5-turbo-0125","Cosine Similarity"]
        '''
        ans_list=[]
        mp_pool=mp.Pool(processes=mp.cpu_count())
        multi_manager=mp.Manager()
        acc_list=multi_manager.list()

        if os.path.isfile(self.acc_datapath):
            checkpoint_data=load_checkpoint(self.acc_datapath)
            logger.info(f"{self.acc_datapath} have {len(checkpoint_data)}")
            acc_list+=checkpoint_data
            ans_list=[i['Answer'] for i in acc_list]
            ground_truth=[i['Ground_Truth'] for i in acc_list]

        eval_acc=acc_metric(self.acc_model)
        logger.info(f"Model {self.acc_model} Start to Calculate accuracy")
        for idx ,(batch,answer,ground_truth) in enumerate(p:=tqdm(self.eval_loader)):
            acc_batch= eval_acc.compute_acc([str(i) for i in answer],[str(i) for i in ground_truth]) # [batch*acc]
            args=[(acc_list,i,float(acc),ans,gt) for i,ans,gt,acc in zip(batch,answer,ground_truth,acc_batch)if ans not in ans_list]

            mp_pool.starmap(self.acc_worker,args)

            p.set_description_str(f"Data Count {len(batch)} ; {len(acc_list)} ; {len(self.eval_loader)}")
            torch.cuda.empty_cache()

        mp_pool.close()
        mp_pool.join()

        Update_file(list(acc_list),self.acc_datapath)
        logger.info(f"Model {self.acc_model} accuracy END {len(acc_list)}")

        return self.acc_datapath


if __name__=="__main__":
    simi_datapath="response_result/20240601/din0s_asqa_gpt-3.5-turbo-0125_vanilla_Long_QA_Cos_sim_No_Shuffle.json"
    key=get_key_()

    acc_datapath=Accuracy("din0s/asqa","gpt-3.5-turbo-0125","Long_QA",key,'vanilla',10,100,"bertscore",simi_datapath).acc_main()
