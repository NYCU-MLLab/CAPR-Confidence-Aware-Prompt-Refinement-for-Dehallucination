from openai import OpenAI
import openai,time,os,threading as th,json
import yaml,torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import anthropic,re
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
                            print(response)
                            continue
                        except Exception as e:
                            # logger.error(f"{th.current_thread().name}Unexpected error: {e} {Instruction} : {input_text} {response}")
                            continue

                except openai.APIStatusError as e:
                    print(f"{th.current_thread().name} code : {e.status_code}_{e}")
                    continue
        else:
            print("Text input empty, please check your input text")
            return None
        return None

class anthropic_GPT:
    def __init__(self,model,api_key,baseline):
        self.model_name=model
        self.api_key=api_key
        self.anthropic_client = anthropic.Anthropic(api_key=self.api_key,)
        self.APIValidation=False
        self.complete_tokens=0
        self.prompt_tokens=0
        self.re_gen_times=1
        self.baseline_task=baseline

    def parser(self,text):
        if self.baseline_task=="confidence":
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

        elif self.baseline_task=="self_polish":
            answer_match = re.search(r'"New_Question": "(.*?)"', text)
            if answer_match:
                answer = str(answer_match.group(1))
            else:
                answer=None
            return {"New_Question":answer}

        elif self.baseline_task=="RaR":
            Expanded_Question = re.search(r'"Expanded_Question": "(.*?)"', text)
            if Expanded_Question:
                expand_q = str(Expanded_Question.group(1))
            else:
                expand_q=None

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

            return {"Expanded_Question":expand_q,"Answer":answer,"Confidence":confidence_score}

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
            print(f"{parser_task} Fail : {result}")
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
            print(f"{parser_task} Fail : {result}")
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
            print(f"{parser_task} Fail : {result}")
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

    elif parser_task=='extract_answer':
        acc=result.get("accuracy",None)
        if acc is not None:
            final_result={
                "Accuracy":acc,
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



    def generate(self,baseline=""):
        # self.api_key=self.api_key['openai']['api_key']
        for _ in range(self.re_gen_times):
            if "gpt" in self.api_name:
                str_response=openai_GPT(self.api_name,self.api_key).ChatGPT_reply(system_prompt=self.system_prompt,Instruction=self.Instruction,question=self.question,input_text=self.input_text,assit_prompt=self.assit_prompt)

                result=str_response

            elif 'claude' in self.api_name:
                claude_client=anthropic_GPT(self.api_name,self.api_key,baseline)
                str_response=claude_client.claude_reply(system_prompt=self.system_prompt,Instruction=self.Instruction,question=self.question,input_text=self.input_text,assit_prompt=self.assit_prompt)
                print(str_response)
                result=claude_client.parser(str_response.replace("\n","").replace("[","").replace("]",""))
                print(result)
                # result=json.loads(str_response)

            if result is not None:
                final_res=ans_parser(self.ans_parser,result)
                if final_res is not None:
                    return final_res
                else:
                    continue
        else:
            print(f"Generate fail exit()\n{self.question}\n{self.Instruction}")
            return None


import textgrad as tg
import re

class text_grad:
    def __init__(self,api_model) -> None:
        self.set_up_keys()
        tg.set_backward_engine(api_model, override=True)

        # Step 1: Get an initial response from an LLM.
        self.model = tg.BlackboxLLM("gpt-4o")

    def set_up_keys(self):
        if os.path.isfile("../api_key.yml"):
            with open("../api_key.yml","r") as f:
                key=yaml.safe_load(f)

        os.environ['OPENAI_API_KEY'] = key['openai']['api_key']


    def parser(self,text):
        answer_match = re.search(r'"answer": "(.*?)"', text)
        if answer_match:
            answer = str(answer_match.group(1))
        else:
            answer=None
        # Regex to capture the confidence score
        confidence_match = re.search(r'"confidence": (\d+\.\d+)', text)
        if confidence_match:
            confidence_score = float(confidence_match.group(1))
        else:
            confidence_score=None
        return {"Answer_before":answer,"Confidence":confidence_score}

    def text_grad_get_response(self,question_str,answer_str):

        question_string = (f"{question_str}"
                        "provide only one 'answer' to the question and one 'confidence' to the answer in json, confidence is a float value between 0 and 1")

        question = tg.Variable(question_string,
                            role_description="question to the LLM",
                            requires_grad=False)

        answer = self.model(question)
        result=self.parser(str(answer).replace("json","").replace("`",""))
        answer.set_role_description("concise and accurate answer to the question")

        # Step 2: Define the loss function and the optimizer, just like in PyTorch!
        # Here, we don't have SGD, but we have TGD (Textual Gradient Descent)
        # that works with "textual gradients".
        optimizer = tg.TGD(parameters=[answer])
        evaluation_instruction = (f"Here's a question:{question_string}."
                                "Evaluate any given answer to this question, "
                                "be smart, logical, and very critical. "
                                "Just provide concise feedback."
                                )
        # TextLoss is a natural-language specified loss function that describes
        # how we want to evaluate the reasoning.
        loss_fn = tg.TextLoss(evaluation_instruction)

        # Step 3: Do the loss computation, backward pass, and update the punchline.
        # Exact same syntax as PyTorch!
        loss = loss_fn(answer)
        loss.backward()
        optimizer.step()
        result['Answer']=self.parser(str(answer).replace("json","").replace("`",""))['Answer_before']
        print("*"*50)
        print(question_str)
        print(answer_str)
        print(str(answer).replace("json","").replace("`",""))
        print(result)
        print("*"*50)
        if None in result.values():
            return None
        else:
            return result




