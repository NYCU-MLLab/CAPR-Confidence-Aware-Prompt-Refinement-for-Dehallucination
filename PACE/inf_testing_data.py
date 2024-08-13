import pandas as pd
import json,re,os,torch,yaml
from qadataset_dataloader import eval_dataloader
from util import get_key_
from tqdm import tqdm
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

key=get_key_()
def Prompter(text):
    return f'''<s>[INST] <<SYS>>You are a expert task to predict Valence and Arousal to the given Sentence.\nValence represents the degree of pleasant(9.000) and unpleasant(0.000) feelings \nArousal represents the degree of excitement(9.000) and calm(0.000).\nOnly Give me floating-points from 0.000 to 9.000. Don't give any Explanation or other words.<</SYS>>Predict Valence and Arousal of a given Sentence: "{text}"\nOutput the answer in JSON in the following format{{"Valence": [Valence in floating-points here],"Arousal": [Arousal in floating-points here]}} Only output JSON don't give any explnation or words.\n:[/INST]'''

def ans_parser(parser_task,result):
    if parser_task=="similarity":
        simi=result.get("similarity",None)
        if simi is not None:
            final_result={"similarity":simi}
            return final_result
        else:
            return None

    elif parser_task=="confidence":
        ans=result.get("Answer",None)
        conf=result.get("Confidence",None)
        explain=result.get("Explanation",None)
        if ans is not None and conf is not None:
            final_result={
                "Answer":ans,
                "Confidence":conf,
                "Explanation":explain
            }
            return final_result
        else:
            return None
    else:
        print(f"Invalid Task {parser_task}")
        return None


def load_checkpoint(datapath:str)->list:
    datares=[]
    if os.path.isfile(datapath):
        with open(datapath,"r") as f:
            datares=json.load(f)
    else:
        print(f"{datapath} not exist")
    return datares

def inference(dataset_path,Pre_train_path=""):
    base_model_name = "meta-llama/Llama-2-7b-chat-hf" #path/to/your/model/or/name/on/hub"

    model = AutoModelForCausalLM.from_pretrained(base_model_name,token=key['hugginface']["token"],torch_dtype=torch.bfloat16,use_cache=True, device_map = 'cuda:0',attn_implementation="flash_attention_2")
    model = PeftModel.from_pretrained(model, Pre_train_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    ## TODO Dataloader
    dataloader=eval_dataloader(dataset_path,1,purpose='compare',tokenizer="",shuffle=False)

    for id, text, text_token in (bar:=tqdm(dataloader)):
        # Ensure input_ids are tensors on the correct device
        # text_token = {key: val.to('cuda:0') for key, val in text_token.items()}
        input_ids = torch.tensor(text_token['input_ids']).to('cuda:0')
        attention_mask = torch.tensor(text_token['attention_mask']).to('cuda:0')
        while True:
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    temperature=1,
                    top_p=0.9,
                    top_k=50,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    max_new_tokens=256,
                )
            response = [tokenizer.decode(r[len(i):],skip_special_tokens=True) for r,i  in zip(outputs, input_ids)]
            parsing_result=ans_parser("confidence",response)

if __name__=="__main__":
    dataset_path="response_result/20240601/din0s_asqa_gpt-3.5-turbo-0125_vanilla_Long_QA.json"
    Pretrained_path=""
    data=inference(dataset_path,Pretrained_path)

