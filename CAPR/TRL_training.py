import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'

from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import nn
# from peft import PeftModel, PeftConfig, get_peft_model
from accelerate import Accelerator
import peft
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from qadataset_dataloader import qadataset_dataloader,eval_dataloader
from torch.nn.parallel import DistributedDataParallel as DDP
# from torchtune.utils import set_seed
from tqdm import tqdm
from trl import PPOConfig,PPOTrainer
import torch.multiprocessing as t_mp
import torch.distributed as dist
from random import randint
from inference import inference
from torch.optim.lr_scheduler import ConstantLR,ExponentialLR,PolynomialLR,SequentialLR,StepLR,LinearLR
import json
from RL_env import Environment,reward_function,rl_writer,Parallel_Environment
import glob,os,torch,yaml
from huggingface_hub import login
import wandb
import torch
from util import get_key_

# torch.cuda.set_device(0)
device = 0 if torch.cuda.is_available() else "cpu"

key=get_key_()

if os.path.isfile("default_config.yaml"):
    with open("default_config.yaml","r") as f:
        ac_config=yaml.safe_load(f)

wandb.login(key=key['wandb']["api_key"])

login(token=key['hugginface']["token"])

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '49527'

    # initialize the process group
    dist.init_process_group("qmao_gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()

def trainer(Batch_accumulate_size, max_epoch, model, tokenizer,Dataloader,generation_kwargs,writer):

    # rank=os.environ['LOCAL_RANK']
    # print(f"Running DDP on rank {rank}.")


    # model = model.to(rank)
    # model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    # old_reward=0
    ## Start To Train With Loaded Dataset
    ppo_trainer=model
    reward_list,query_tensors_list,response_tensors_list=[],[],[]
    # stream = torch.cuda.current_stream(rank)
    ppostep_size=8*Batch_accumulate_size
    reward_list=[]
    query_tensors_list=[]
    response_tensors_list=[]
    show_result=True
    example_list=[]

    for epoch in (t:=tqdm(range(max_epoch), "epoch: ")):

        for prompt,instruct,instruct_token,ans,ground_Truth,Confidence,Document in (bar:=tqdm(Dataloader,leave=True)):
            bar.set_postfix_str("get Instruction")
            instruct_token.input_ids=list(map(lambda x:torch.tensor(x),instruct_token.input_ids))
            query_tensors=instruct_token.input_ids
            #### Get response from SFTModel
            bar.set_postfix_str("get Response")
            response_tensors = ppo_trainer.generate(query_tensor=query_tensors, return_prompt=True, **generation_kwargs)
            instruct_token['response'] = tokenizer.batch_decode(response_tensors)
            instruct_token['query'] = instruct

            response = [tokenizer.decode(r[len(i):],skip_special_tokens=True) for r,i  in zip(response_tensors, instruct_token.input_ids)]
            assert len(prompt)==len(response)
            show_index=randint(0,len(response)-1)
            ##################### Show Result
            if show_result:
                example={
                    'Epoch':epoch,
                    'Sample_id':show_index,
                    'Ori_instruct':prompt[show_index]['Instruction'],
                    'Question':prompt[show_index]['Question'],
                    'Question':prompt[show_index]['Question'],
                    'New_instruct':response[show_index]
                }

            ## replace generated Instruction
            for idx,p_instruc in enumerate(response):
                prompt[idx]['Instruction']=str(p_instruc)
                prompt[idx]['system_prompt']="This is a QA task, please answer the Question base on the Instruction to the question and confidence to the Answer in json."
                prompt[idx]['input_text']="\nOnly give me one Answer and Confidence according to response format in json, don't give me any other words.\n\nresponse format:\n{'Answer':[ONLY Your final Answer here],\n'Confidence':[Your final Confidence here]}"

            ## Environment Get Answer and Confidence
            bar.set_postfix_str("get Environment")

            result_batch=Parallel_Environment(prompt,key,'gpt-3.5-turbo-0125')


            bar.set_postfix_str("get Reward")
            # print(result_batch)
            #### Compute reward score
            Reward,ece,origin_ece,acc,pace_conf,conf = reward_function(result_batch,ground_Truth,Document)
            ##################### Show Result
            if show_result:
                try:
                    get_result=result_batch[show_index]['Answer']
                except:
                    get_result=result_batch[show_index]

                example.update({
                    'result':get_result,
                    'ground_Truth':ground_Truth[show_index],
                    'Accuracy':acc[show_index].item(),
                    'Confidence':conf[show_index].item(),
                    'PACE Confidence':pace_conf[show_index].item(),
                    'ECE':ece[show_index].item()
                })

            if show_result:
                example_list.append(example)
                with open(f'{writer.determint}.json','w+') as f:
                    json.dump(example_list,f)

            reward_list+=Reward
            query_tensors_list+=query_tensors
            response_tensors_list+=response_tensors
            bar.set_description_str(f"Epoch {epoch},{len(reward_list)},grad step:{ppostep_size} acc {torch.mean(torch.stack(acc)).item():.5f}")
            #### Run PPO step
            if len(reward_list)>=ppostep_size:
                bar.set_postfix_str("PPO Step")
                stats = ppo_trainer.step(query_tensors_list[:ppostep_size], response_tensors_list[:ppostep_size], reward_list[:ppostep_size])

                ### Record the result
                writer.get([stats['ppo/val/error']],measure='mean',key='Val_error')
                writer.get([stats['ppo/learning_rate']],measure='mean',key='learning_rate')
                writer.get([stats['ppo/loss/value']],measure='mean',key='loss_value')
                writer.get([stats['ppo/loss/total']],measure='mean',key='loss_total')
                writer.get([stats['ppo/loss/policy']],measure='mean',key='loss_policy')
                writer.get([stats['ppo/policy/approxkl']],measure='mean',key='ApproxKL')
                writer.get([stats['ppo/policy/policykl']],measure='mean',key='policykl')
                writer.get(Reward,measure='mean',key='reward')
                writer.get(acc,measure='mean',key='Accuracy')
                writer.get(ece,measure='mean',key='ECE')
                writer.get(pace_conf,measure='mean',key='Pace_Verbalized_Confidence')
                writer.get(conf,measure='mean',key='Verbalized_Confidence')
                writer.write()
                #####################
                show_result=True
                reward_list=[]
                query_tensors_list=[]
                response_tensors_list=[]
                #####################
            else:
                show_result=False

        ppo_trainer.save_pretrained(f"Agent_weight/PPO_Agent_{writer.determint}_{epoch}_{stats['ppo/loss/total']:.4f}")
        if epoch in [3,5,7]:
            input(f"Epoch {epoch}, Press Enter to continue training, You can check the training result now...")

        ppo_trainer.log_stats(stats, instruct_token, Reward,columns_to_log=["query", "response"])
        torch.save(stats,f"{writer.data_folder}/{epoch}_state.pth")

        #### Save model
        Agent_addres=f"Agent_weight/PPO_Agent_{writer.determint}_{epoch}_{stats['ppo/loss/total']:.4f}"
        ppo_trainer.save_pretrained(Agent_addres)
        t.set_description_str(f"Epoch {epoch}:")
    return Agent_addres

def main():

    # torch.cuda_set_device(1)

    Training_Config={
        "dataset_path":f'response_result/20240601/triviaQA_gpt-3.5-turbo-0125_vanilla_QA.json', ## Training Data
        'deliminator':"06122032_vanilla_f1_r1_trivia_withPACE", ## Save_File deliminator
        'max_epoch': 8, ## Training Epoch
        'trian_batch_size':8, ## Training Batch Size
        'Batch_accumulate_size':32 ## Training Batch Accumulate Size min : 128, Max: 64
    }

    # pretrained_model_path=""
    pretrained_model_path=f"Agent_weight/PPO_Agent_{Training_Config['deliminator']}_2_0.0090"

    writer=rl_writer(Training_Config['deliminator'])

    os.makedirs('Agent_weight', exist_ok=True)
    torch.cuda.empty_cache()
    n_gpus = torch.cuda.device_count()
    # assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus

    ## Greater than 16 and the power of 8


    ###
    config = PPOConfig(
        model_name="meta-llama/Llama-2-7b-chat-hf",
        world_size=2,
        batch_size=Training_Config['trian_batch_size']*Training_Config['Batch_accumulate_size'],
        mini_batch_size=8,
        optimize_cuda_cache=True,
        early_stopping=True,
        target_kl=0.9,
        init_kl_coef=0.01,
        adap_kl_ctrl=True,
        gradient_accumulation_steps=16,
        ppo_epochs=8,
        is_peft_model=True,
        ratio_threshold= 10.0,
        max_grad_norm=1,
        compare_steps=1
        # log_with = "wandb",
    )

    # set_seed(config.seed)

    # current_device=1
    # device_map = {"": Accelerator().local_process_index}
    device_map = {"": 0}

    peft_config = peft.AdaptionPromptConfig(adapter_len = 10, adapter_layers = 30)

    if os.path.isdir(pretrained_model_path):
        model = AutoModelForCausalLMWithValueHead.from_pretrained(pretrained_model_path,token=key['hugginface']["token"],torch_dtype=torch.bfloat16,use_cache=True,device_map=device_map)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path,token=key['hugginface']["token"])
        print("="*50+"Load From Pretrained !!!"+"="*50)
    else:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name, peft_config = peft_config, token=key['hugginface']["token"],torch_dtype=torch.bfloat16,use_cache=True,device_map=device_map)
        tokenizer = AutoTokenizer.from_pretrained(config.model_name,token=key['hugginface']["token"])
        print("="*50+"Load From Huggingface !!!"+"="*50)

    optim_confg=[{
        'params':model.v_head.parameters(),
        'lr':1e-4
    },{
        'params':model.pretrained_model.parameters(),
        'lr':1e-3
    }
                 ]
    ## tokenizer init
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    Dataloader=eval_dataloader(dataset_path=Training_Config['dataset_path'], batch_size=Training_Config['trian_batch_size'], purpose='refine',tokenizer=tokenizer,shuffle=True).trainloader

    optim = torch.optim.AdamW(optim_confg, eps = 1e-4,weight_decay=0.01)

    # scheduler1 = StepLR(optim, step_size=9, gamma=0.9)
    ## Overall Iter
    Overalliter=len(Dataloader)*Dataloader.batch_size*Training_Config['max_epoch']//config.batch_size
    ##
    print(f"Total Step: {Overalliter}")
    scheduler2 = PolynomialLR(optim,  total_iters=Overalliter, power=2)
    # scheduler3 = ExponentialLR(optim, gamma=0.9)
    warmup = LinearLR(optim, start_factor=1e-2,end_factor=1,total_iters=int(Overalliter*0.02)+1)

    main_schedualer=SequentialLR(optim, schedulers=[warmup, scheduler2], milestones=[warmup.total_iters])

    # optim = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    ppo_trainer = PPOTrainer(
        model=model,
        config=config,
        tokenizer=tokenizer,
        optimizer=optim,
        lr_scheduler=main_schedualer,
        )

    # ds_plugin = ppo_trainer.accelerator.state.deepspeed_plugin

    # model,Dataloader,optim ,lr_scheduler= accelerator.prepare(model,Dataloader,optim,lr_scheduler)

    generation_kwargs = {
        "min_length": -1,
        'temperature': 1,
        "max_length": 256,
        # "max_new_tokens": 96, # before : 128
        "top_k": 50,
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        'no_repeat_ngram_size':4
        }

    Agent_addres=trainer(Training_Config['Batch_accumulate_size'], Training_Config['max_epoch'], ppo_trainer, tokenizer,Dataloader,generation_kwargs,writer)

    # inference(Training_Config['dataset_path'],f"din0s_asqa_{Training_Config['deliminator']}_Vanilla.json",Agent_addres)
    # Show_mean_result(f"din0s_asqa_{Training_Config['deliminator']}_Vanilla.json")

if __name__=="__main__":

    main()

