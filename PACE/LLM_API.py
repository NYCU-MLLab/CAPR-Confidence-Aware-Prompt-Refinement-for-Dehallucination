from openai import OpenAI
import openai,time,os,threading as th,json
from util import setup_logger
import yaml,torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import anthropic,re

logger = setup_logger('log/gpt_class.log')
import multiprocessing as mp

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class openai_GPT:
    def __init__(self,model,api_key):
        self.model_name=model
        self.api_key=api_key
        self.openai_client = OpenAI(
            api_key=self.api_key,)
        self.APIValidation=False
        self.complete_tokens=0
        self.prompt_tokens=0
        self.re_gen_times=1

    def  ChatGPT_reply(self,system_prompt='',Instruction='',question='',input_text='',temperature=0,max_tokens=4096,assit_prompt=""):
        if input_text:
            for _ in range(self.re_gen_times):
                try:
                    response=self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages= [
                        {"role": "system", "content":f"{str(system_prompt)}"},
                        {"role": "user", "content":f"{str(Instruction)} {str(question)} {str(input_text)}"},
                        {"role": "assistant", "content": f"{str(assit_prompt)}"}
                        ],

                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={ "type": "json_object" }
                    )
                    if dict(response).get('choices',None) is not None:
                        self.APIValidation=True
                        try:
                            claim_text= json.loads(response.choices[0].message.content)
                            return claim_text
                        except json.JSONDecodeError as e:
                            # logger.error(f"{th.current_thread().name}JSON decoding failed: {e} {Instruction} : {input_text} {response}")
                            continue
                        except Exception as e:
                            # logger.error(f"{th.current_thread().name}Unexpected error: {e} {Instruction} : {input_text} {response}")
                            continue

                except openai.APIStatusError as e:
                    logger.error(f"{th.current_thread().name} code : {e.status_code}_{e}")
                    continue
        else:
            logger.debug("Text input empty, please check your input text")
            return None
        return None

class anthropic_GPT:
    def __init__(self,model,api_key):
        self.model_name=model
        self.api_key=api_key
        self.anthropic_client = anthropic.Anthropic(api_key=self.api_key,)
        self.APIValidation=False
        self.complete_tokens=0
        self.prompt_tokens=0
        self.re_gen_times=1

    def  claude_reply(self,system_prompt='',Instruction='',question='',input_text='',temperature=0,max_tokens=4096,assit_prompt=""):
        if input_text:
            for _ in range(self.re_gen_times):
                # try:
                    response = self.anthropic_client.messages.create(
                        model=self.model_name,
                        max_tokens=1000,
                        temperature=0,
                        system=f"{str(system_prompt)}\n{str(assit_prompt)}",
                        messages=[

                            {
                                "role": "user",
                                "content": [
                                    {"type": "text",
                                    "text":f"{str(Instruction)} {str(question)} {str(input_text)}"
                                    }
                                            ]
                            }
                        ]
                    )
                    claim_text=response.content[0].text
                    return claim_text
                # except:
                #     return None


def LlamaChatCompletion(model_name, prompt, max_tokens):

    model_name = "daryl149/llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(input_ids=input_ids,
                            max_new_tokens=max_tokens,return_dict_in_generate=True, output_scores=True, output_hidden_states=True)

    tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # pdb.set_trace()
    return outputs



def LLAMA3_response(model,tokenizer,prompt_batch)->str: ## Change to batch


    # f'''<|begin_of_text|><|start_header_id|>{prompt["system_prompt"]}<|end_header_id|>

    #     {{ system_prompt }}<|eot_id|><|start_header_id|>{prompt["user_prompt"]}<|end_header_id|>

    #     {{ user_msg_1 }}<|eot_id|><|start_header_id|>{prompt["assistant_prompt"]}<|end_header_id|>

    #     {{ model_answer_1 }}<|eot_id|>'''

      # 确保所有输入都是一个批次
    def LLAMA3_message(p):
        messages = [
            {"role": "system", "content": f"{p['system_prompt']},{p['assit_prompt']}"},
            {"role": "user", "content": p['user_prompt']+p['input_text']},
        ]
        return messages

    batch_input_ids = []
    for messages in list(map(LLAMA3_message,prompt_batch)):
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
        )
        # print(input_ids.squeeze(0).shape)
        batch_input_ids.append(input_ids)

    token=tokenizer(batch_input_ids, padding=True, return_tensors="pt",max_length=4096,truncation=True).to(device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        **token,
        max_new_tokens=4096,
        eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    # 解析生成的响应
    responses = []
    for i, output in enumerate(outputs):
        response = output[token.input_ids[i].shape[-1]:]
        res = tokenizer.decode(response, skip_special_tokens=True)
        responses.append(res)

    return responses


def ans_parser(parser_task,result):
    if parser_task=="similarity":
        simi=result.get("similarity",None)
        if simi is not None:
            try:
                final_result={"similarity":float(simi)}
                return final_result
            except:
                return None
        else:
            logger.info(f"{parser_task} Fail : {result}")
            return None

    elif parser_task=="confidence":
        ans=result.get("Answer",None)
        conf=result.get("Confidence",None)
        explain=result.get("Explanation",None)
        if ans is not None and conf is not None:
            try:
                final_result={
                    "Answer":str(ans),
                    "Confidence":float(conf),
                    "Explanation":str(explain)
                }

                return final_result
            except:
                return None
        else:
            logger.info(f"{parser_task} Fail : {result}")
            return None

    elif parser_task=="multi_step_confidence":
        if result.get("Answer",None) is not None and result.get("Confidence",None) is not None:
            multi_result={}
            for k,v in result.items():
                if f"Step" in k:
                    multi_result[k]=v
            final_result={
                "Explanation":multi_result,
                "Confidence":result.get("Confidence",0),
                "Answer":result.get("Answer","")
                }
            return final_result
        else:
            logger.info(f"{parser_task} Fail : {result}")
            return None

    elif parser_task=="refine":
        pp=result.get("New_Prompt",None)
        if pp is not None :
            final_result={
                "New_Prompt":pp
                }
            return final_result
        else:
            # logger.info(f"{parser_task} Fail : {result}")
            return None

    elif parser_task=='self_polish':
        pp=result.get("New_Question",None)
        if pp is not None :
            final_result={
                "New_Question":pp
                }
            return final_result
        else:
            # logger.info(f"{parser_task} Fail : {result}")
            return None

    elif parser_task=='RaR':
        pp=result.get("Expanded_Question",None)
        ans=result.get("Answer",None)
        Conf=result.get("Confidence",None)
        if pp is not None:
            final_result={
                "Expanded_Question":pp,
                'Answer':ans,
                'Confidence':Conf
                }
            return final_result
        else:
            # logger.info(f"{parser_task} Fail : {result}")
            return None

class GPT_API:
    def __init__(self,api_name,api_key,ans_parser,prompt):
        '''
        This is API response Function
        Need input api_name and api_key
        Then system_prompt for system character
        user_promt+input_text for input
        assis_prompt is the assistent character
        return
        1. claim text (response text):str
        2. complete tokens : output tokens
        3. prompt token : input tokens
        '''
        self.api_name=api_name
        self.api_key=api_key
        self.ans_parser=ans_parser
        self.re_gen_times=3
        self.system_prompt=prompt["system_prompt"]
        self.Instruction=prompt["Instruction"]
        self.input_text=prompt["input_text"]
        self.question=prompt["Question"]
        self.assit_prompt=prompt["assit_prompt"]
        self.api=OpenAI(api_key=api_key)

    def parser(self,text):
        answer_match = re.search(r'"Answer": "(.*?)"', text)
        if answer_match:
            answer = str(answer_match.group(1))
        else:
            answer=None
        # Regex to capture the confidence score
        confidence_match = re.search(r'"Confidence": (\d+\.\d+)', text)
        if confidence_match:
            confidence_score = float(confidence_match.group(1))
        else:
            confidence_score=None
        return {"Answer":answer,"Confidence":confidence_score}

    def generate(self):
        # self.api_key=self.api_key['openai']['api_key']
        for _ in range(self.re_gen_times):
            if "gpt" in self.api_name:
                str_response=openai_GPT(self.api_name,self.api_key).ChatGPT_reply(system_prompt=self.system_prompt,Instruction=self.Instruction,question=self.question,input_text=self.input_text,assit_prompt=self.assit_prompt)

                result=str_response

            elif 'claude' in self.api_name:
                str_response=anthropic_GPT(self.api_name,self.api_key).claude_reply(system_prompt=self.system_prompt,Instruction=self.Instruction,question=self.question,input_text=self.input_text,assit_prompt=self.assit_prompt)
                # print(str_response)
                result=self.parser(str_response.replace("\n","").replace("[","").replace("]",""))
                # print(result)
                # result=json.loads(str_response)

            if result is not None:
                final_res=ans_parser(self.ans_parser,result)
                if final_res is not None:
                    return final_res
                else:
                    continue
        else:
            logger.error(f"Generate fail exit()\n{self.question}\n{self.Instruction}")
            return None

if __name__=="__main__":
    pass

