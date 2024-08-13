import argparse
import multiprocessing as mp
from util import *
from evaluate_confidence_score import evaluate_score,Savefig
import os,json,torch
from datetime import datetime
import yaml
from PACE_Confidence import Confidence
from PACE_Similarity import Similarity
from PACE_accuracy import Accuracy
# import wandb
os.environ["TOKENIZERS_PARALLELISM"] = "True"
# activation_time=datetime.now().strftime("%Y%m%d")
activation_time='20240601'
os.makedirs('./log/',exist_ok=True)
logger = setup_logger(f'log/response_{activation_time}.log')
# wandb.init(project='PACE',resume='allow')

tasks = {
    'din0s/asqa': 'Long_QA',
    'natural_questions': 'QA',
    'triviaQA': 'QA'
    }
accuracy_model_mapping={
    'din0s/asqa': 'rougeL',
    'natural_questions': 'f1',
    'triviaQA': 'f1'
}
api_model_key_mapping={
    "gpt-4-turbo":"openai",
    "gpt-3.5-turbo-0125":"openai",
    'claude-3-5-sonnet-20240620':"claude"
}

ans_parser_dict= {
    'vanilla': 'confidence',
    'cot': 'confidence',
    "multi_step":"multi_step_confidence",
    }

def pipeline(qa_dataset,api_model,tasks,Stretagy,data_prompt_amount,train_batch_size,key,eval_batch_size,sim_model,acc_model,ans_parser,shuffle,lambda_value):

    Confident_datapath=Confidence(qa_dataset,api_model,tasks[qa_dataset],key,data_prompt_amount,train_batch_size,Stretagy,activation_time,ans_parser).confidence_main()
    logger.info(f"{Confident_datapath} Finish")
    simi_datapath=Similarity(qa_dataset,api_model,tasks[qa_dataset],key,Stretagy,data_prompt_amount,eval_batch_size,sim_model,Confident_datapath,shuffle).simi_main()
    logger.info(f"{simi_datapath} Finish")
    acc_datapath=Accuracy(qa_dataset,api_model,tasks[qa_dataset],key,Stretagy,data_prompt_amount,eval_batch_size,acc_model,simi_datapath).acc_main()
    logger.info(f"{acc_datapath} Finish")
    logger.info(f"*******************************************************************")
    evaluate_score(api_model,qa_dataset,Stretagy,sim_model,acc_model,activation_time,tasks[qa_dataset],acc_datapath,lambda_value,shuffle)


def main():
    torch.cuda.empty_cache()
    os.makedirs(f'response_result/{activation_time}', exist_ok=True)
    os.makedirs('log',exist_ok=True)
    key=get_key_()
    # 'natural_questions','din0s/asqa',"triviaQA",
    datasets = ["din0s/asqa"]
    # strategies = ['vanilla']
    strategies = ['vanilla','cot','multi_step']
    sim_models = 'Cos_sim'

    ## API model
    api_model = 'gpt-3.5-turbo-0125'
    # api_model = 'gpt-4-turbo'
    # api_model = 'claude-3-5-sonnet-20240620'

    api_key=key[api_model_key_mapping[api_model]]['api_key']

    shuffle=False
    data_count = 500
    train_batch_size = 1
    eval_batch_size = 0 ## No use
    lambda_value=0.5
    for qa_dataset in datasets:
        for strategy in strategies:
            logger.info(f"Start With {mp.cpu_count()} CPUs :  {qa_dataset} {strategy} {accuracy_model_mapping[qa_dataset]}")
            pipeline(qa_dataset,api_model,tasks,strategy,data_count,train_batch_size,api_key,eval_batch_size,sim_models,accuracy_model_mapping[qa_dataset],ans_parser_dict[strategy],shuffle,lambda_value)

    # shuffle_str="shuffle" if shuffle else "No_shuffle"
    # Savefig(f"{activation_time}_{shuffle_str}",api_model,datasets,sim_models,accuracy_model_mapping,strategies)
# File_name,api_model,dataset="din0s/asqa",sim_models="Cos_sim",acc_model="rougeL"
def shell_ver():
    parser = argparse.ArgumentParser(description="Put Parameter in Generating")
    parser.add_argument("--qa_dataset", type=str,default="din0s/asqa",choices=["din0s/asqa","ChilleD/StrategyQA","natural_questions",'gsm8k'], help="QA Dataset Option")
    parser.add_argument("--api_model", type=str,default="gpt-3.5-turbo-0125",choices=["gpt-3.5-turbo-0125","llama2","llama3"], help="Model name for API")
    parser.add_argument("--Stretagy", type=str, default='vanilla',choices=['vanilla','cot'], help="Stretagy for Prompt")
    parser.add_argument("--With_rag", type=bool, default=False,help="if use RAG or not for prompt or not")
    parser.add_argument("--sim_model", type=str, default='gpt-3.5-turbo-0125',choices=['gpt-3.5-turbo-0125','Cos_sim'], help="Similarity method")
    parser.add_argument("--task", type=str, default='QA',choices=['QA','Long_QA'], help="Task")
    parser.add_argument("--acc_model", type=str, default='bertscore',choices=['bertscore',"wer",'exact_match','bool_acc','f1','recall',"bool_acc"], help="ACC method")
    # data_prompt_amount,train_batch_size,eval_batch_size
    parser.add_argument("--data_prompt_amount", type=int, default=200, help="Calculate Data Amount")
    parser.add_argument("--train_batch_size", type=int, default=100, help="Confidence Task loader Batch size")
    parser.add_argument("--eval_batch_size", type=int, default=50, help="Similarity Task loader Batch size")
    args = parser.parse_args()

    logger.info(f"Start With {mp.cpu_count()} CPUs \n {args}")

# def single_test():
#     os.makedirs(f'response_result/{activation_time}', exist_ok=True)
#     os.makedirs('log', exist_ok=True)
#     if os.path.isfile("api_key.yml"):
#         with open("api_key.yml","r") as f:
#             key=yaml.safe_load(f)

#     datasets = ['natural_questions','din0s/asqa']
#     strategies = ['vanilla', 'cot',"multi_step"]
#     acc_models = ["EM"]
#     api_model = 'gpt-3.5-turbo-0125'
#     sim_model = 'snli_max'

#     data_count = 1
#     train_batch_size = 5
#     eval_batch_size = 2
#     With_rag=False

#     for qa_dataset in datasets:
#         for strategy in strategies:
#             for acc_model in acc_models:
#                 logger.info(f"Start With {mp.cpu_count()} CPUs :  {qa_dataset} {strategy} {acc_model}")
#                 pipeline(qa_dataset,api_model,tasks,strategy,data_count,train_batch_size,With_rag,key,eval_batch_size,sim_model,acc_model,ans_parser_dict[strategy])

#     Savefig(activation_time)


if __name__=="__main__":
    '''
    Dataset :
        Long Form QA : "din0s/asqa","Eli5","QAMPARI"
        Short Form QA : "natural_questions",
        Summerization Task : "abisee/cnn_dailymail",
    Evaluate Metric: ECE, AUROC,
    Accuracy : bertscore, f1.
    Similarity ( Document / Answer ): Cosine Similarity.
    Prompt Stretagy :
        1. Vanilla
        2. Chain-of-Thought
        3. multi_step
        4. Top-k
    '''
    main()
    # single_test()
    # overall_= Get_Cost(file_path='response_result/')
    # logger.info(f"Overall Cost: {overall_} USD {overall_*30} NTD")
    Savefig(f"20240601_No_shuffle",'gpt-3.5-turbo-0125',['din0s/asqa',"triviaQA"],"Cos_sim",accuracy_model_mapping,['vanilla','cot','multi_step'])



