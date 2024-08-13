
from LLM_API import *
from qadataset_dataloader import qadataset_dataloader
from rouge_score import rouge_scorer
import os
from tqdm import tqdm
from util import *
from prompt_strategy import prompter
import yaml,os,json,re
import multiprocessing as mp
from sklearn.metrics import roc_auc_score,roc_curve,auc
from copy import copy,deepcopy
import textgrad as tg

key=get_key_()

#  gpt-3.5-turbo-0125, gpt-4-turbo
# "claude-3-5-sonnet-20240620"

api_model='claude-3-5-sonnet-20240620'
api_key=key['claude']['api_key']

# api_model='gpt-4-turbo'
# api_key=key['openai']['api_key']

def Get_auroc(accuracy,confidence_scores):
    y_true=np.where(np.array(accuracy) < 0.9,0,1)
    fpr1, tpr1, thresholds1 = roc_curve(y_true, np.array(confidence_scores))
    roc_auc=auc(fpr1, tpr1)

    return roc_auc

def rewrite_worker(idx,original_question,ground_truth,documnet,baseline,acc_metric):
    result={}
    if baseline =="vanilla":
        prompt=question_to_prompt([original_question],'QA','vanilla')
        Answer_result=GPT_API(api_model,api_key,'confidence',prompt).generate('confidence')
        if Answer_result is not None:
            Accuracy=float(ans_scorer(Answer_result['Answer'],ground_truth,acc_metric))
            result={
                'Id':idx,
                'Original_question':original_question,
                'Documnet':documnet,
                'Ground_truth':ground_truth,
                'Answer':Answer_result['Answer'],
                'Confidence':Answer_result['Confidence'],
                'Accuracy':Accuracy,
            }

    elif baseline =="self_polish":
        ###### self polish iterate refine question
        old_refine_question=deepcopy(original_question)
        question_list=[old_refine_question]
        old_Accuracy=0
        Final_result={}
        for idx in (p:=tqdm(range(3),leave=True,position=1)):
            prompt=question_to_prompt([old_refine_question],'self_polish','self_polish')
            new_question=GPT_API(api_model,api_key,'self_polish',prompt).generate(baseline)


            # if Answer_result is not None:
            #     Accuracy=float(ans_scorer(Answer_result['Answer'],ground_truth,acc_metric))
            #     p.set_postfix_str(f"Update {Accuracy}")
            #     if Accuracy>old_Accuracy:
            #         old_Accuracy=Accuracy
            #         Final_result={
            #             'Answer':Answer_result['Answer'],
            #             'Confidence':Answer_result['Confidence'],
            #             'Accuracy':float(Accuracy),
            #         }
            #         p.set_description_str(f"Max {Final_result['Accuracy']}")

            # old_refine_question=new_question['New_Question']
            # question_list.append(new_question['New_Question'])

        new_question_prompt=question_to_prompt([new_question["New_Question"]],'QA','vanilla')
        Final_result=GPT_API(api_model,api_key,'confidence',new_question_prompt).generate("confidence")
        Accuracy=float(ans_scorer(Final_result['Answer'],ground_truth,acc_metric))
        if Final_result:
            result={
                'Id':idx,
                'Original_question':original_question,
                'New_Question':new_question['New_Question'],
                'Question_history':question_list,
                'Documnet':documnet,
                'Ground_truth':ground_truth,
                'Answer':Final_result['Answer'],
                'Confidence':float(Final_result['Confidence']),
                'Accuracy':float(Accuracy),
            }
        else:
            print(Final_result)

    elif baseline =="RaR":
        prompt=question_to_prompt([original_question],'QA','RaR')
        Answer_result=GPT_API(api_model,api_key,'RaR',prompt).generate(baseline)

        if Answer_result is not None:
            Accuracy=float(ans_scorer(Answer_result['Answer'],ground_truth,acc_metric))
            result={
                'Id':idx,
                'Original_question':original_question,
                'Expanded_question':Answer_result['Expanded_Question'],
                'Answer':Answer_result['Answer'],
                'Ground_truth':ground_truth,
                'Confidence':float(Answer_result['Confidence']),
                'Accuracy':float(Accuracy),
                'Documnet':documnet,
            }
    if result:
        return result
    return None

def evaluate_result(datapath):
    if os.path.isfile(datapath):
        with open(datapath,'r') as f:
            data=json.load(f)

        acc=np.array([float(i['Accuracy']) for i in data])
        conf=np.array([float(i['Confidence']) for i in data])
        ece_score=np.abs(acc-conf)
        print(f"{datapath}")
        print(f"Accuracy mean :{np.mean(acc)}")
        print(f"ECE mean :{np.mean(ece_score)}")
        print(f"Auroc mean :{Get_auroc(acc,conf)}")
    else:
        print(f"Not Exist {datapath}")

def main(baseline,datapath,acc_metric='f1'):

    if baseline in ["vanilla","self_polish","RaR"]:
        train_dataloader=qadataset_dataloader(dataset_path="triviaQA",split='validation',batch_size=1,shuffle=False).trainloader
        share_list=[]

        for idx,(batch,Ground_truth) in enumerate(progress:=tqdm(train_dataloader,position=0)):
            original_question=[i[0] for i in batch]
            document=[search_wikipedia_byurl(i[1]) if i[2] else i[1] for i in batch]
            for idx,(q,gt,doc) in enumerate(zip(original_question,Ground_truth,document)):
                kkresult=rewrite_worker(idx,q,gt,doc,baseline,acc_metric)
                if kkresult is not None:
                    share_list.append(kkresult)

            if share_list:
                progress.set_description_str(f"Processing {len(share_list)} batch acc {np.mean(np.array([i['Accuracy'] for i in share_list]))}")
                progress.set_postfix_str(f"list length: {len(share_list)}")
            if len(share_list)>=50 or idx >=50:
                break

            with open(datapath,'w+') as f:
                json.dump(list(share_list),f)

    elif baseline in ["textgrad"] and 'gpt' in api_model:
        train_dataloader=qadataset_dataloader(dataset_path="triviaQA",split='validation',batch_size=1,shuffle=False).trainloader
        share_list=[]
        for idx,(batch,Ground_truth) in enumerate(progress:=tqdm(train_dataloader)):
            original_question=[i[0] for i in batch]
            document=[search_wikipedia_byurl(i[1]) if i[2] else i[1] for i in batch]

            for q,doc,a in zip(original_question,document,Ground_truth):

                Answer_result=text_grad(api_model).text_grad_get_response(q,a)
                if Answer_result is not None:
                    Accuracy1=float(ans_scorer(Answer_result['Answer'],a,acc_metric))
                    Accuracy2=float(ans_scorer(Answer_result['Answer_before'],a,acc_metric))
                    if Accuracy1 > Accuracy2:
                        share_list.append({
                            'Id':idx,
                            'Original_question':q,
                            'Answer':Answer_result['Answer'],
                            'Answer_before':Answer_result['Answer_before'],
                            'Ground_truth':a,
                            'Confidence':float(Answer_result['Confidence']),
                            'Accuracy':float(Accuracy1),
                            'Documnet':doc,
                        })
                    else:
                        share_list.append({
                            'Id':idx,
                            'Original_question':q,
                            'Answer':Answer_result['Answer'],
                            'Answer_before':Answer_result['Answer_before'],
                            'Ground_truth':a,
                            'Confidence':float(Answer_result['Confidence']),
                            'Accuracy':float(Accuracy2),
                            'Documnet':doc,
                        })
                    with open(datapath,'w+') as f:
                        json.dump(list(share_list),f)
                else:
                    progress.set_description_str(f"Fail {Answer_result}")
                if share_list:
                    progress.set_description_str(f"{idx} Processing {len(share_list)} batch acc {np.mean(np.array([i['Accuracy'] for i in share_list]))}")


            if len(share_list)>=50 or idx >=50:
                break

if __name__=="__main__":
    baseline_list=["vanilla","self_polish","RaR"]
    for baseline in baseline_list:
        datapath=f'baseline_result/trivia_{api_model}_{baseline}_detail.json'
        main(baseline,datapath,'extract_answer')
        evaluate_result(datapath)

    # print(ans_scorer("48 hours","48 Hrs.","extract_answer"))

