import argparse,torch
from LLM_API import GPT_API
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

class Confidence:
    def __init__(self,dataset_path,api_model,task,apikey,data_prompt_amount,train_batch_size,stretagy,activation_time,ans_parser=''):
        ## generate setting
        self.dataset_path=dataset_path
        self.update_freq=1
        self.data_prompt_amount=data_prompt_amount
        self.train_batch_size=train_batch_size
        ## Document tokenizer setting
        self.tokens_per_part=96
        self.doc_tokenizer_name='xbert'
        self.task=task
        self.ans_parser=ans_parser
        ## init
        self.prompter=prompter()
        self.api_model=api_model
        self.api_key=apikey
        self.stretagy=stretagy

        self.Confident_datapath=f"response_result/{activation_time}/{self.dataset_path.replace('/','_')}_{self.api_model}_{self.stretagy}_{self.task}.json"
        logger.info(f"Confident_datapath: {self.Confident_datapath}")
        self.train_dataloader=qadataset_dataloader(dataset_path=dataset_path,split='train',batch_size=self.train_batch_size).trainloader

    def confidence_worker(self,Share_list,batch,label,batch_size):
        # if question not in share_dict:
        if batch is not None:
            ques,doc,isurl=batch
            if ques is not None and batch_size>0:
                question=change_list_dict_to_str(ques)
                if question not in [i['Question'] for i in Share_list]:
                    if isurl:
                        document=search_wikipedia_byurl([doc])
                    else:
                        document=[doc]
                    # print(type(document))
                    assert isinstance(document,list)
                    if document != ["No knowledge"]:
                        if document:
                            document=chunk_document("".join(document),self.tokens_per_part,self.doc_tokenizer_name)
                        self.prompter.setup_task(self.task)
                        ## (prompt)
                        p=self.prompter.get_prompt(query=[question],document=document,stretagy=self.stretagy)
                        if self.stretagy =="topk": ## topk
                            response_candidiate=[]
                            for _ in range(3):
                                result=GPT_API(self.api_model,self.api_key,self.ans_parser,p).generate()
                                response_candidiate.append(result)
                        else:
                            result=GPT_API(self.api_model,self.api_key,self.ans_parser,p).generate()

                        if result is not None:
                            if self.dataset_path=="ChilleD/StrategyQA":
                                ans =False if "No" in ans else True

                            Share_list.append({
                                'Prompt':p,
                                'Question':question,
                                'Document':document,
                                'Explanation':result["Explanation"],
                                'Answer':result["Answer"],
                                'Confidence':float(result["Confidence"]),
                                'Ground_Truth':label,
                                'Complete_tokens':0,
                                'Prompt_tokens':0
                                                    })
                        else:
                            logger.info(f"{mp.current_process().name} {question} return None")
                else:
                    logger.info(f"{mp.current_process().name} {question} Exist")
        else:
            logger.info(f"{mp.current_process().name} batch is None")

    def confidence_main(self):
        # Iterate over the DataLoader
        # mp_pool=mp.Pool(processes=mp.cpu_count())
        mp_pool=mp.Pool(processes=1)
        multi_manager=mp.Manager()
        Share_list=multi_manager.list()
        # Create a managed lock
        average_conf=0
        old_data_size = 0
        ## Load from checkpoint
        if os.path.isfile(self.Confident_datapath):
            datares= load_checkpoint(self.Confident_datapath)
            Share_list+=datares
            logger.info(f"{self.Confident_datapath} have:  {len(Share_list)}")
            old_data_size=len(Share_list)

        logger.info(f"{self.stretagy} Prompt Stretagy get conf and answer from {self.api_model}")


        for idx,(batch,Ground_truth) in enumerate(progress:=tqdm(self.train_dataloader)):
            ## [[question, Doc ,isurl],ground_truth,wer]
            ### MultiProcess Parallel
            if len(Share_list)>=self.data_prompt_amount:
                break
            if Ground_truth is not None and None not in batch:
                progress.set_description_str(f"{self.stretagy};{self.api_model}")
                args=[(Share_list,b,g,len(batch)) for b,g in zip(batch,Ground_truth)]
                mp_pool.starmap(self.confidence_worker,args)

                progress.set_postfix_str(f"{old_data_size}")
                if idx%self.update_freq==0:
                    Update_file(list(Share_list),self.Confident_datapath)
                    old_data_size=len(Share_list)


            Update_file(list(Share_list),self.Confident_datapath)

        logger.info(f"{self.stretagy} Prompt END {self.api_model}")
        mp_pool.close()
        mp_pool.join()


        self.clean_conf_data(self.Confident_datapath)
        return self.Confident_datapath


    def clean_conf_data(self,target_datapath):
        datares= load_checkpoint(target_datapath)

        for idx,v in enumerate(datares):
            datares[idx]['Answer']=change_list_dict_to_str(v['Answer'])
        assert type(datares[idx]['Document']) is list
        logger.info(f"{target_datapath} Load {len(datares)} is clean")
        Update_file(list(datares),target_datapath)



if __name__=="__main__":
    if os.path.isfile("api_key.yml"):
        with open("api_key.yml","r") as f:
            key=yaml.safe_load(f)

    train_dataloader=qadataset_dataloader(dataset_path="natural_questions",split='train',batch_size=1).loader
    p_=prompter()
    p_.setup_task("QA")

    for idx,(batch,Ground_truth) in enumerate(train_dataloader):
        question=batch[0][0]
        document=[batch[0][1]]
        document=chunk_document("".join(document),96,'xbert')
        prompt=p_.get_prompt(query=[question],document=document,stretagy="multi_step")
        llm_api=GPT_API("gpt-3.5-turbo-0125",key,"multi_step_confidence",prompt)
        result=llm_api.generate()
        print(result)

