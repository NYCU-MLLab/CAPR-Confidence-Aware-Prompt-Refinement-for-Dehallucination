import torch
from transformers import BertTokenizer
import logging
from logging.handlers import RotatingFileHandler
from sentence_transformers import SentenceTransformer, util
import requests
import wikipediaapi

from tqdm import tqdm
from urllib.parse import unquote
from evaluate import load
from bert_score import BERTScorer
from bs4 import BeautifulSoup
import os,json,glob
from datetime import datetime
import numpy as np
import multiprocessing as mp
import random,yaml
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt


def get_key_():
    if os.path.isfile("../api_key.yml"):
        with open("../api_key.yml","r") as f:
            key=yaml.safe_load(f)
            print("Key Get !!")
        return key
    else:
        print("Key FAIL !!")
        return None

def shuffle_theans(ll):
    # random.seed(42)
    random.shuffle(ll)
    # Define the range of indices to shuffle
    start_ratio=0
    end_ratio=1
    start_index = int(len(ll) * start_ratio)
    end_index = int(len(ll) * end_ratio) # Ending index of the part to shuffle (exclusive)

    # Shuffle only the specified part of the list
    sublist = ll[start_index:end_index]
    random.shuffle(sublist)
    ll[start_index:end_index] = sublist

    return ll

def setup_logger(logfile):
    os.makedirs("log",exist_ok=True)
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)  # 设置日志记录器的级别为DEBUG

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        file_handler = RotatingFileHandler(logfile, maxBytes=1024*1024*5, backupCount=5)
        file_handler.setLevel(logging.DEBUG)  # 设置文件处理器的级别为DEBUG
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

        # 添加控制台处理器，将日志消息输出到控制台
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # 设置控制台处理器的日志级别为INFO
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

activation_time=datetime.now().strftime("%Y_%m_%d")
logger = setup_logger(f'log/response_{activation_time}.log')

def load_checkpoint(datapath:str)->list:
    datares=[]
    if os.path.isfile(datapath):
        with open(datapath,"r") as f:
            datares=json.load(f)
    else:
        logger.info(f"{datapath} not exist")
    return datares

def change_list_dict_to_str(target):
    if isinstance(target, dict):
        return " ".join([f"{k} {v}" for k,v in target.items()])
    if isinstance(target,list):
        return "".join(target)
    return target

model_huggingface = {
    'bert': "bert-base-uncased",
    'xbert': 'efederici/sentence-bert-base',
}
# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained(model_huggingface['xbert'])

def chunk_document(text, tokens_per_part, model_name='xbert'):
    TOKEN_LIMIT = tokens_per_part
    MAX_MODEL_LENGTH = 512

    tokenizer.model_max_length =200000
    # Split the text into manageable chunks
    tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=False)['input_ids'][0]

    # Create chunks with a limit of MAX_MODEL_LENGTH
    token_chunks = [tokens[i:i+MAX_MODEL_LENGTH] for i in range(0, len(tokens), MAX_MODEL_LENGTH)]

    final_chunks = []

    for token_chunk in token_chunks:
        # Further split each chunk if it exceeds TOKEN_LIMIT
        for i in range(0, len(token_chunk), TOKEN_LIMIT):
            final_chunks.append(token_chunk[i:i+TOKEN_LIMIT])

    # Convert each chunk back to text
    chunks_text = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in final_chunks]

    return chunks_text


def split_text_into_fixed_length_parts(text, tokens_per_part, model_name='xbert'):

    model_huggingface={
        'bert':'bert-base-uncased',
        'xbert':'efederici/sentence-bert-base',
    }
    # Initialize the tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_huggingface[model_name])
    # Tokenize the text
    tokens = tokenizer(text,padding=True,truncation=True,return_tensors='pt').input_ids
    print(tokens)
    # Initialize the list to hold each part
    parts = []
    # Calculate the number of full parts
    full_parts = len(tokens) // tokens_per_part
    # Create parts with exactly tokens_per_part tokens
    for i in range(full_parts):
        start_index = i * tokens_per_part
        end_index = start_index + tokens_per_part
        part_tokens = tokens[start_index:end_index]
        # Convert token list to string and add to the parts list
        parts.append(tokenizer.convert_tokens_to_string(part_tokens))

    # Handle the remaining tokens, if any
    if len(tokens) % tokens_per_part:
        remaining_tokens = tokens[full_parts * tokens_per_part:]
        parts.append(tokenizer.convert_tokens_to_string(remaining_tokens))

    return parts


def Update_file(datares:list,datapath:str):
    with open(datapath,"w+") as f:
        json.dump(datares,f)
        logger.info(f"{mp.current_process().name}, Write Data: {len(datares)} {datapath}")

def scrape_external_knowledge_by_request(urls: list) -> list:
    find_limit = None

    if urls:
        url = urls.pop(0)

        if url is not None:
            try:
                response = requests.get(url)

                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    paragraphs = soup.find('div', class_='mw-parser-output')

                    if paragraphs:
                        paragraphs = paragraphs.find_all('p', limit=find_limit)
                        summary = "".join([paragraph.text for paragraph in paragraphs]).split("\n")

                        if summary:
                            return summary[1:len(summary)-1]
                        else:
                            return ["No knowledge"]
                    else:
                        return ["No knowledge"]
                else:
                    print(f"Failed to retrieve the page: Status code: {response.status_code}")
                    return ["No knowledge"]
            except Exception as e:
                print(f"An error occurred: {e}")
                return ["No knowledge"]
        else:
            return ["No knowledge"]
    else:
        print("No URL provided")
        return ["No knowledge"]


class simi_metric:
    def __init__(self, metric: str):
        self.metric = metric
        self.load_metric()

    def load_metric(self):
        if self.metric in ['Cos_sim']:
            self.eval_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
            # 'sentence-transformers/all-MiniLM-L6-v2'
            # "sentence-transformers/all-MiniLM-L12-v2"

    def compute_similarity(self, pred: list, ans: list)->list:

        if self.metric == 'Cos_sim':
            try:
                embedding_1 = self.eval_model.encode(pred, convert_to_tensor=True)
                embedding_2 = self.eval_model.encode(ans, convert_to_tensor=True)
                print(embedding_1.shape)
                print(embedding_2.shape)
                result = util.pytorch_cos_sim(embedding_1, embedding_2)
                return result.cpu().numpy()[0].astype(float)
            except:
                return [0]
class acc_metric:
    def __init__(self,metric:str):
        self.metric=metric
        self.load_metric()

    def load_metric(self,load_layer=False):

        if self.metric in ['bertscore','f1','recall']:
            with tqdm(disable=True):
                self.eval_model = BERTScorer(
                        model_type="bert-base-uncased",  # Model type
                        num_layers=9,  # Number of layers to use
                        all_layers=False,  # Whether to use all layers
                        idf=False,  # Whether to use IDF scaling
                        batch_size=64,  # Batch size
                        lang=None,  # Language of the texts, auto-detect based on model if None
                        rescale_with_baseline=False,  # Whether to rescale scores with baselines
                        device='cuda:0'  # Specify the CUDA device
                    )

        elif self.metric in ['wer','exact_match']:
            self.eval_model=load(self.metric)

        elif self.metric in ['rouge1','rouge2','rougeL']:
            self.eval_model = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def compute_acc(self,pred:list,ans:list)->list:
        if self.metric =='bertscore':
            P, R, F1 =self.eval_model.score(pred,ans)
            return P.numpy()
        elif self.metric =='f1':
            P, R, F1 =self.eval_model.score(pred,ans)
            return F1.numpy()
        elif self.metric =='recall':
            P, R, F1 =self.eval_model.score(pred,ans)
            return R.numpy()

        elif self.metric=='wer':
            return [1-self.eval_model.compute(predictions=[i],references=[j]) for i,j in zip(pred,ans)]

        elif self.metric=='exact_match':
            return [self.eval_model.compute(predictions=[i],references=[j])['exact_match'] for i,j in zip(pred,ans)]

        elif self.metric=='bool_acc':
            return self.bool_acc(pred,ans)

        elif self.metric in ['rouge1','rouge2','rougeL']:
            return [self.eval_model.score(p, a)[self.metric].fmeasure for p,a in zip(pred,ans)]

    def bool_acc(self,pred:list,ans:list):
        correct_matches = [1 if str(p).lower() == str(r).lower() else 0 for p, r in zip(pred, ans)]
        # print(correct_matches)
        # Calculate accuracy
        # accuracy = sum(correct_matches) / len(pred)  # Convert to percentage
        return correct_matches

    def compute_em_score(self,predictions:list, ground_truths:list)-> float: ## acc
        """
        Compute the Exact Match (EM) score.

        Parameters:
            predictions (list of str): The list of predicted answers.
            ground_truths (list of str): The list of actual answers.

        Returns:
            float: The EM score calculated as the percentage of exact matches.
        """
        # Check if both lists have the same number of elements
        if len(predictions) != len(ground_truths):
            raise ValueError("The length of predictions and ground truths must be the same.")

        # Calculate the number of exact matches
        exact_matches = sum([1 for pred, truth in zip(predictions, ground_truths) if pred == truth])

        # Compute the EM score
        em_score = exact_matches / len(predictions)
        return em_score

def Get_Cost(file_path='response_result'):
    ### Count_tokens
    res={}
    overall_=0
    if os.path.isfile(file_path):
        res[file_path]={'Complete_tokens':0,'Prompt_tokens':0}
        datares=load_checkpoint(file_path)

        for i in datares:
            if "Complete_tokens" in i:
                res[file_path]['Complete_tokens']+=i['Complete_tokens']
            if "Prompt_tokens" in i:
                res[file_path]['Prompt_tokens']+=i['Prompt_tokens']

    elif os.path.isdir(f"{file_path}"):
        for datapath in os.listdir(file_path):
            for i in glob.glob(f'{file_path}/{datapath}/*.json'):
                overall_+=Get_Cost(i)
                # res['compete_toekn']+=Count_Tokens(i)['compete_toekn']
                # res['Prompt_tokens']+=Count_Tokens(i)['Prompt_tokens']

    ### Get Cost
    for k,v in res.items():
        resTotal_Spent =v['Complete_tokens']*1.5/1000000+v['Prompt_tokens']*0.5/1000000
        overall_+=resTotal_Spent
        if resTotal_Spent>0:
            print(f"{k.split('/')[-1]}:{v['Complete_tokens']} / {v['Prompt_tokens']}")
            print(f"Sprent {resTotal_Spent} USD {resTotal_Spent*30} NTD\n")
    return overall_



def search_wikipedia_by_keyward(keyword):
    headers = {'User-Agent': 'CoolBot/0.0 (https://example.org/coolbot/; coolbot@example.org)'}
    # Initialize the Wikipedia API
    wiki_wiki = wikipediaapi.Wikipedia(user_agent='CoolBot/0.0 (https://example.org/coolbot/; coolbot@example.org)', language='en')

    # Search for the keyword
    page = wiki_wiki.page(keyword)

    # Check if the page exists
    if page.exists():
        # Print the title and summary of the page
        print(f"Title: {page.title}")
        print(page.ATTRIBUTES_MAPPING)
        print(page.text)
        # print(f"Summary: {page.summary[:500]}...")  # Print the first 500 characters of the summary
    else:
        print(f"No page found for '{keyword}'")


def search_wikipedia_byurl(url:list)->list:
    # print(url)
    def get_page_from_url(url):
        # Extract the page name from the URL
        decoded_url = unquote(url)
        if "wikipedia.org/wiki/" in url:
            page_name = decoded_url.split("wikipedia.org/wiki/")[1]
            return True,page_name
        else:
            print("Invalid Wikipedia URL")
            result=scrape_external_knowledge_by_request([url])
            return False,result
    if url:
        ifexits,page_name = get_page_from_url(url.pop())
        if page_name is None:
            return ["No knowledge"]

        if ifexits:
            wiki_wiki = wikipediaapi.Wikipedia(user_agent='CoolBot/0.0 (https://example.org/coolbot/; coolbot@example.org)', language='en')
            page = wiki_wiki.page(page_name)

            if page.exists():
                return [page.text]
            else:
                # print(f"No page found for {page_name}")
                return ["No knowledge"]
        else:
            return page_name

def show_histogram_graph(vector,title,activate_time,stretagy="",sim="",datafile_name="",label=[]):
    os.makedirs(f"picture/histogram/{activate_time}",exist_ok=True)
    # Plot histogram
    # plt.figure(figsize=(4, 3))
    colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
    random.shuffle(colors)
    for i,j in zip(vector,label):
        plt.hist(i, bins=100, density=True, alpha=0.7, color=colors[random.randint(0,len(colors)-1)], edgecolor='black',label=j)
    # Add a title and labels
    plt.title(f'{title}')
    plt.xlabel('Count')
    plt.ylabel('Frequency')
    # plt.ylim(0,1)
    plt.xlim(0,1)
    # Add a grid
    plt.grid(True)
    plt.legend(loc='upper right')
    # Show plot
    plt.savefig(f"picture/histogram/{activate_time}/{datafile_name}.png")
    # plt.show()
    plt.clf

def Update_Fig(activate_time,shuffle=True):
    isshuffle_str="shuffle" if shuffle else "No_shuffle"
    datapaht=f"./response_result/Evaluate_Result_{activate_time}_{isshuffle_str}.json"
    with open(datapaht,'r') as f:
        data=json.load(f)

    os.makedirs("picture/histogram",exist_ok=True)
    simi_models=["Cos_sim"]
    datasets=["natural_questions","din0s/asqa"]
    api_model='gpt-3.5-turbo-0125'
    acc_model='bertscore'

    stretagy=["vanilla",'cot',"multi_step"]

    for sim in simi_models:
        for dataset in datasets:
            for dd in data:
                for k in stretagy:
                    conf_list,Final_conf_list,simi_list=[],[],[]
                    if dd['dataset']==dataset and dd['sim_model']==sim and dd['Stratagy']==k:
                        print(f"Load Sucess {dataset} {k} {sim}")
                        conf_list.append(dd['Conf'])
                        Final_conf_list.append(dd['Pace_Conf'])
                        simi_list.append(dd['Simi'])
                        dataset_path=dataset.replace("/","_")
                        show_histogram_graph([conf_list],activate_time=activate_time,title=f"{k} {sim} Confidence",stretagy=k,sim=f"{sim}",datafile_name=f"{dataset_path}_{isshuffle_str}_{sim}_{k}_Confidence",label=[f"{isshuffle_str}"])

                        show_histogram_graph([simi_list],activate_time=activate_time,title=f"{k} {sim} Similarity",stretagy=k,sim=f"{sim}",datafile_name=f"{dataset_path}_{isshuffle_str}_{sim}_{k}_Similarity",label=[f"{isshuffle_str}"])

                        show_histogram_graph([Final_conf_list],activate_time=activate_time,title=f"{k} {sim} Final Confidence",stretagy=k,sim=f"{sim}",datafile_name=f"{dataset_path}_{isshuffle_str}_{sim}_{k}_PACE_Confidence",label=[f"{isshuffle_str}"])


if __name__=="__main__":

    # a="https://en.wikipedia.org/wiki/The%20Wizard%20of%20Oz%20%281939%20film%29"
    # b="https://en.wikipedia.org/wiki/List%20of%20Bunk%27d%20episodes"
    # c="https://en.wikipedia.org/wiki/2003%20FIFA%20Women%27s%20World%20Cup"
    # d="https://en.wikipedia.org/wiki/Human%20Development%20Index'"
    e="https://en.wikipedia.org/wiki/Australian%20gold%20rushes"
    # a= 'https://wonderopolis.org/wonder/will-a-watermelon-grow-in-your-belly-if-you-swallow-a-seed'
    a= 'https://en.wikipedia.org/wiki/Ku_Klux_Klan'
    # result=search_wikipedia_byurl([a])
    # print(result)

    ref=['today is friday?',"Obama is cool","This is today"]
    pred=['today is friday']*3
    # ref=['today is friday?']
    # sim_me=simi_metric('Cos_sim')
    # Score=sim_me.compute_similarity(pred,ref)
    # print(Score)
    # print(np.mean(Score))
    # print(max(Score))
    eval_me=acc_metric('f1')
    print(torch.tensor(eval_me.compute_acc(pred,ref)))
    # keyword = "Artificial Intelligence"
    # search_wikipedia_by_keyward("Human Development Index")


