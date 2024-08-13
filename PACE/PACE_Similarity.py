import torch
from LLM_API import *
from prompt_strategy import prompter
from qadataset_dataloader import eval_dataloader
import multiprocessing as mp
from util import *
from evaluate_confidence_score import evaluate_score
import json,os
from tqdm import tqdm
from infer_snli_cls import snli_similarity,Load_snli_model
import os,json
import random
from datetime import datetime
import yaml
logger = setup_logger(f'log/response_{datetime.now().strftime("%Y%m%d")}.log')

class Similarity:
    def __init__(self,dataset_path,api_model,task,key,stretagy,data_prompt_amount,eval_batch_size,sim_model,Confident_datapath,shuffle):
        ## generate setting
        self.dataset_path=dataset_path
        self.data_prompt_amount=data_prompt_amount
        self.eval_batch_size=eval_batch_size
        self.task=task
        ## init
        self.sim_model=sim_model
        self.Stretagy=stretagy
        self.prompter=prompter()
        self.api_model=api_model
        self.api_key=key
        self.Confident_datapath=Confident_datapath
        ## Setting Shuffle
        self.shuffle=shuffle
        self.shuffle_str="Shuffle" if self.shuffle else "No_Shuffle"

        self.eval_loader=eval_dataloader(self.Confident_datapath,self.eval_batch_size,'compare').loader
        self.simi_datapath=self.Confident_datapath.replace(f'.json',f'_{self.sim_model}_{self.shuffle_str}.json')

        logger.info(f"Similarity datapath: {self.simi_datapath}")

        self.mp_progress_setting={
            "gpt-3.5-turbo-0125":mp.cpu_count(),
            "Cos_sim":1,
            "snli":20,
        }
        if not os.path.isfile(self.Confident_datapath):
            logger.error(f"{self.Confident_datapath} Do not Exists")
            exit()

    def simi_worker(self,share_list,batch,answer,groundtruth):
        ## simi : ans:str,  Document:list
        if batch[0] not in [i["Question"] for i in share_list]:
            similist=[]
            c_token,p_token=0,0
            for doc in (p:=tqdm(batch[1])):
                self.prompter.setup_task('similarity')
                ## Documment and Answer's similarity
                assert isinstance(doc,str)
                p1=self.prompter.get_prompt(query=[str(answer)],document=[doc],stretagy='similarity',with_rag=False)## (prompt)
                llm_api=GPT_API(self.api_model,self.api_key,"similarity",**p1)
                res,indi_complete_tokens,indi_Prompt_tokens=llm_api.generate()
                if res:
                    similist.append(float(res.get("similarity",0)))
                    c_token+=indi_complete_tokens
                    p_token+=indi_Prompt_tokens
                    p.set_postfix_str(f"Similarity {float(res.get('similarity',0))}")
                    p.set_description_str(f"gpt3.5")

            if not similist:
                similist=[0]

            share_list.append({
                        'Prompt':batch[4],
                        'Question':batch[0],
                        'Answer':answer,
                        'Confidence':batch[2],
                        'Explanation':batch[3],
                        'Ground_Truth':groundtruth,
                        "Doc_Ans_simi":similist,
                        'Document':batch[1],
                        'Complete_tokens':c_token,
                        'Prompt_tokens':p_token
                    })
            # logger.info(f"{len(share_list)}/{batch_size} similist count :{len(similist)},tokens: {c_token}/{p_token}")
        # else:
            # logger.info(f"{question} Exist")

    def simi_main(self):
        '''
        Get Document and Response Similarity Score for Confidence Calibration
        Option : ["gpt-3.5-turbo-0125","Cos_sim"]
        '''
        logger.info("Shffle ANS") if self.shuffle else logger.info("Not Shffle ANS")


        checkpoint_data=[]

        if os.path.isfile(self.Confident_datapath):

            if self.sim_model=="gpt-3.5-turbo-0125":
                mp_pool=mp.Pool(processes=self.mp_progress_setting[self.sim_model])
                share_list=mp.Manager().list()
                share_list+=checkpoint_data

                ## GPT for similarity
                logger.info(f"Model {self.sim_model} Start to Calculate similarity")

                for idx,batch,ans,ground_truth in enumerate(p:=tqdm(self.eval_loader)):
                    if self.shuffle:
                        ans=shuffle_theans(ans)
                    p.set_description_str(f"Data Count {len(batch)} ; {len(share_list)} ; {len(self.eval_loader)}")
                    mp_pool.starmap(self.simi_worker,[(share_list,i,answ,ground) for i,answ,ground in zip(batch,ans,ground_truth)])

                    Update_file(list(share_list),self.simi_datapath)
                    torch.cuda.empty_cache()
                    # break

                mp_pool.close()
                mp_pool.join()

            elif self.sim_model=='Cos_sim':
                self.simi_metric=simi_metric("Cos_sim")
                share_list=[]

                for idx ,(batch,ans,ground_truth) in enumerate(p:=tqdm(self.eval_loader)):
                    if self.shuffle:
                        ans=shuffle_theans(ans)
                    for i,answ,ground in tqdm(zip(batch,ans,ground_truth)):
                        # similist=[]
                        similist=list(map(float,self.simi_metric.compute_similarity([answ]*len(i[1]),i[1])))

                        p.set_description_str(f"{len(similist)}; {sum(similist)/len(similist)}; {max(similist)}")
                        # print(type(similist),similist)

                        if not similist:
                            similist=[0]

                        share_list.append({
                                    'Prompt':i[4],
                                    'Question':i[0],
                                    'Answer':answ,
                                    'Confidence':i[2],
                                    'Explanation':i[3],
                                    'Ground_Truth':ground,
                                    "Doc_Ans_simi":similist,
                                    'Document':i[1],
                                    'Complete_tokens':0,
                                    'Prompt_tokens':0
                                })
                    # if len(share_list)>len(checkpoint_data):
                    Update_file(list(share_list),self.simi_datapath)
                        # checkpoint_data=share_list
                else:
                    Update_file(list(share_list),self.simi_datapath)

                del self.simi_metric
                torch.cuda.empty_cache()

            ## Snli Similarity for similarity Score between Response and Document
            elif 'snli' in self.sim_model:
                share_list=[]
                label_dict={
                    0:"entailment",
                    1:"neutral",
                    2:"contradiction",
                }
                Bmodel,Btokenizer=Load_snli_model("bert_model_weight/bert_snli_classifier_eval_M3_0.92.pth")
                for idx,(batch,ans,ground_truth) in enumerate(p:=tqdm(self.eval_loader)):
                    if self.shuffle:
                        ans=shuffle_theans(ans)
                    for i,answ,ground in zip(batch,ans,ground_truth):
                        similist=[]
                        ## Shuffle
                        # Set the random seed for reproducibility
                        for doc in (q:=tqdm(i[1])):
                            predicted_label,probabilities,outputs=snli_similarity(Bmodel,Btokenizer,answ,doc)
                            ## take entailment probability
                            if label_dict[predicted_label]=="entailment":
                                simi_score=float(outputs[0][0])
                            elif label_dict[predicted_label]=="contradiction":
                                simi_score=float(outputs[0][0])
                            else:
                                simi_score=float(0.0)
                            ##
                            similist.append(simi_score)
                            # logger.info(f"{mp.current_process().name}, {label_dict[predicted_label]} Doc_Ans_simi:{simi_score}")
                            q.set_postfix_str(f"{label_dict[predicted_label]} Doc_Ans_simi:{simi_score}")
                            q.set_description_str(f"{len(similist)}; {sum(similist)/len(similist)}; {max(similist)}")

                        if not similist:
                            similist=[0]

                        share_list.append({
                                    'Prompt':i[4],
                                    'Question':i[0],
                                    'Answer':answ,
                                    'Confidence':i[2],
                                    'Explanation':i[3],
                                    'Ground_Truth':ground,
                                    "Doc_Ans_simi":similist,
                                    'Document':i[1],
                                    'Complete_tokens':0,
                                    'Prompt_tokens':0
                                })
                    # if len(share_list)>len(checkpoint_data):
                    Update_file(list(share_list),self.simi_datapath)
                        # checkpoint_data=share_list
                else:
                    Update_file(list(share_list),self.simi_datapath)
                del Bmodel,Btokenizer
                torch.cuda.empty_cache()

            logger.info(f"Model {self.sim_model} simi END {len(share_list)}")
            return self.simi_datapath
        else:
            logger.error(f"{self.Confident_datapath} Do not Exists")
            return ""

