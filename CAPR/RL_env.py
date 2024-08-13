from LLM_API import GPT_API
from util import *
import yaml,os,torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import multiprocessing as mp
import json
from netcal.metrics import ECE

# key=get_key_()

key_mapping={
    'gpt-3.5-turbo-0125':'openai',
    'gpt-4-turbo':'openai',
    'claude-3-5-sonnet-20240620':'claude',
}

def generate_worker(share_list,key,prompt,id,model_name):
    # print(prompt)
    model=GPT_API(model_name,key[key_mapping[model_name]]['api_key'],"confidence",prompt)
    result=model.generate("confidence")
    if result is not None:
        share_list[id]=result
    else:
        share_list[id]=None

def Parallel_Environment(prompt:list,key,model_name='gpt-3.5-turbo-0125'):
    '''
    This is the Parallel Environment Function
    '''
    mp_pool=mp.Pool(processes=mp.cpu_count())
    multi_manager=mp.Manager()
    Share_list=multi_manager.dict()

    args=[]
    for id,p in enumerate(prompt):
        if p is not None and isinstance(p,dict):
            args.append((Share_list,key,p,id,model_name))
    mp_pool.starmap(generate_worker,args)
    mp_pool.close()
    mp_pool.join()
    return [Share_list[i] for i in sorted(Share_list)]
    # result,c_tokens,p_tokens=llm_api.generate()



def Environment(prompt,key,model_name='gpt-3.5-turbo-0125'):
    '''
    This is the Environment Function
    '''
    if model_name=='gpt-3.5-turbo-0125':
        llm_api=GPT_API("gpt-3.5-turbo-0125",key,"confidence",prompt)
        return llm_api.generate()
    elif model_name=='llama3':
        pass


def reward_function(result_batch,Ground_truth,Document):
    '''
    Batch Input
        answer: Batch* 1
        Ground_truth: Batch* 1
        conf_batch: Batch* 1
        Document_List: Batch* K
    '''

    eval_acc=acc_metric('extract_answer')
    simi=simi_metric("Cos_sim")
    lambda_value=0.7
    ## Balance Between ECE and ACC
    ece_acc_ratio=1.0
    ## ece_acc_ratio*-ece+(1.0-ece_acc_ratio)*acc
    assert len(result_batch)==len(Ground_truth)==len(Document)
    ## result_batch dict(confidece,Answer) / None
    conf_list=[]
    Final_conf_list=[]
    acc_list=[]
    hit=0
    # print(len(result_batch))
    for result,ground,doc in (r:=tqdm(zip(result_batch,Ground_truth,Document),leave=False)):

        with torch.set_grad_enabled(False):
            if result is not None:
                answer=str(result['Answer'])
                ## Follow f1 score
                conf_batch=torch.tensor(float(result['Confidence']))
                acc_batch=torch.tensor(eval_acc.compute_acc([answer],[ground])).squeeze()


                # conf_batch[acc_batch==0.0]=0.0
                simi_scores = torch.tensor(simi.compute_similarity([answer],doc))
                simi_score = torch.max(simi_scores)
                simi_score[simi_score>=0.5]=1.0
                Final_conf=conf_batch*lambda_value+simi_score*(1-lambda_value)
                # Final_conf[acc_batch==1.0]=1.0
                print(f"{answer}, {ground}, {acc_batch}, {conf_batch},{simi_score},{Final_conf}")
                ## ECC = (ACC - CONF)
                ## Reward = -ECC + ACC
                # eceloss=get_ece(Final_conf,acc_batch)
                conf_list.append(conf_batch)
                acc_list.append(acc_batch)
                Final_conf_list.append(Final_conf)
                hit+=1
            else:
                conf_list.append(torch.tensor(1.0))
                Final_conf_list.append(torch.tensor(1.0))
                acc_list.append(torch.tensor(0.0))

        r.set_postfix_str(f"Hit Rate {hit/len(result_batch):.2f}")

    # ece_score=get_ece(conf_list,acc_list)
    # ece_list=get_ece(Final_conf_list,acc_list)
    ece_list=torch.abs(torch.stack(acc_list)-torch.stack(conf_list))
    Final_ece_list=torch.abs(torch.stack(acc_list)-torch.stack(Final_conf_list))
    reward_list=[(1.0-ece_acc_ratio)*acc+ece_acc_ratio*(-ece) for acc,ece in zip(acc_list,Final_ece_list)]

    r.set_description_str(f"reward {max(reward_list).item():.4f}/{min(reward_list).item():.4f},acc {max(acc_list).item():.4f}/{min(acc_list).item():.4f},ece {max(ece_list):.4f}/{min(ece_list):.4f}")

    return reward_list,Final_ece_list,ece_list,acc_list,Final_conf_list,conf_list

def get_ece(y_confs,y_true):
    y_confs=np.array([i.item() for i in y_confs])
    y_true=np.array([i.item() for i in y_true])
    accuracy = np.mean(y_true)
    y_true=np.where(y_true < accuracy,0,1) ## change to binary ## init 0.59
    # ECE
    n_bins = 10
    # diagram = ReliabilityDiagram(n_bins)
    ece = ECE(n_bins)
    ece_score = ece.measure(y_confs, y_true)

    return torch.tensor(ece_score)


class rl_writer:
    def __init__(self,determint="") -> None:
        self.determint=determint
        self.data_folder=f"PPO_State_{determint}"
        os.makedirs(self.data_folder,exist_ok=True)
        self.data_value_writer={}

    def get(self,data_value:list,measure="mean",key=""):

        data_value=list(map(torch.tensor,data_value))

        if key not in self.data_value_writer:
            self.data_value_writer[key]=[]

        if measure=="mean":
            data_value_after=torch.mean(torch.stack(data_value)).item()
            self.data_value_writer[key].append(data_value_after)
        elif measure=="all":
            data_value_after=[i.item() for i in torch.stack(data_value)]
            self.data_value_writer[key]+=data_value_after

    def write(self):
        if self.data_value_writer:
            for k,v in self.data_value_writer.items():
                self.save_result(v,k)

        self.Moving_average()

    def Moving_average(self):
        data_size=10
        ratio=0.9
        for k in ['reward','Accuracy','ECE']:
            data_path=f'{self.data_folder}/{k}.json'

            if os.path.isfile(data_path):
                with open(data_path,'r') as f:
                    data=json.load(f)
                movin_avg=sum(data[:data_size])/data_size
                k1=[]
                for i in data:
                    movin_avg=ratio*movin_avg+(1-ratio)*i
                    k1.append(movin_avg)

                plt.plot(range(len(k1)),k1,label=k,marker='')
                plt.legend()
                plt.savefig(f"{data_path.replace('.json','mvavg.png')}")
                plt.clf()

    def save_result(self,sample,title):
        plt.figure(figsize=(6, 4))
        plt.plot(list(range(1, len(sample) + 1)), sample, marker='', linestyle='-', color='b')
        plt.xlabel('Step')
        plt.ylabel('value')
        plt.grid(True)
        # plt.title(f"{title}")
        plt.legend([f'{title}'])
        plt.savefig(f'{self.data_folder}/{title}.png')
        plt.close()
        with open(f"{self.data_folder}/{title}.json","w") as f:
            json.dump(sample,f)

if __name__=="__main__":
    pass
