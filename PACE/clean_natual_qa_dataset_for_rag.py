from datasets import load_dataset
from torch.utils.data import DataLoader,Dataset
from datetime import datetime
from util import *
from tqdm import tqdm
import multiprocessing as mp
import os,re
activation_time=datetime.now().strftime("%Y_%m_%d")
logger = setup_logger(f'log/response_{activation_time}.log')


import re

def remove_html_tags(html):
    """
    Removes all HTML tags from a given HTML string.

    :param html: str, the HTML content as a string.
    :return: str, the text with all HTML tags removed.
    """
    # Regular expression pattern for removing HTML tags
    clean_text = re.sub(r'<[^>`]+>', '', html)
    clean_text = clean_text.replace('<', '').replace('>', '')
    return clean_text

def collect_fn(batch):
    res=[]
    for i in batch:
        gg=False
        short_ans=[]
        long_ans=[]

        wiki_doc=i['document']['tokens']['token']
        document_text=remove_html_tags(" ".join(wiki_doc))
        short_ans_candidate=i['annotations']['short_answers']

        for short_ans_c in short_ans_candidate:

                if short_ans_c['start_token'] :
                    gg=True
                    if isinstance(short_ans_c['start_token'],list):
                        for s,e in zip(short_ans_c['start_token'],short_ans_c['end_token']):
                            short_ans.append(", ".join(wiki_doc[s:e]))

                    elif isinstance(short_ans_c['start_token'],int):
                        if short_ans_c["start_token"]!=-1:
                            short_ans.append(", ".join(wiki_doc[short_ans_c['start_token']:short_ans_c['end_token']]))

        # long_ans_candidate=i['long_answer_candidates']
        long_ans_candidate=i['annotations']['long_answer']
        for long_ans_c in long_ans_candidate:
            if long_ans_c["candidate_index"]!=-1:
                if long_ans_c['start_token'] :
                    if isinstance(long_ans_c['start_token'],list):
                        for s,e in zip(long_ans_c['start_token'],long_ans_c['end_token']):
                            long_ans.append(" ".join(wiki_doc[s:e]))

                    elif isinstance(long_ans_c['start_token'],int):
                            long_ans.append(" ".join(wiki_doc[long_ans_c['start_token']:long_ans_c['end_token']]))
        if gg:
            res.append([i['question']['text'],document_text,False,short_ans,long_ans])
    return res

def worker(qa_list,question:str,document:str,isurl:bool,short_anwser:list,long_answer:list):
    short_anwser=list(map(remove_html_tags,short_anwser))
    long_answer=list(map(remove_html_tags,long_answer))
    if not question in [i["Question"] for i in qa_list] and short_anwser:
        qa_list.append({
                "Question":remove_html_tags(question),
                "Document":document,
                "isurl":isurl,
                "Short_answer":short_anwser,
                "long_answer":long_answer
            })

def Update_file(datares:list,datapath:str):
    with open(datapath,"w+") as f:
        json.dump(datares,f)
        # print(f"Write {datapath}")

# def create_dataset():
#     dataset_path="natural_qa_dataet/natualQA.json"
#     qa_dict={}
#     for i in glob.glob("natural_qa_dataet/natual_qa_*.json"):
#         old_data=load_checkpoint(f"{i}")
#         qa_dict.update(old_data)
#         res=[]
#         for k in qa_dict.values():
#             res.append(k)
#         Update_file(res,i.replace(".json","_clean.json"))

def main():
    dataset_path='natural_questions'
    dataset = load_dataset(dataset_path)
    data_size=len(dataset['train'])//500
    natual_DataLoader=DataLoader(dataset['train'], batch_size=100, collate_fn=collect_fn,shuffle=False)

    manerger=mp.Manager()
    mp_pool=mp.Pool(processes=mp.cpu_count())
    qa_list=manerger.list()

    data_count=0
    for idx,batch in enumerate(p:=tqdm(natual_DataLoader)):
        qadatapath=f'natural_qa_dataet/natual_qa_{data_count}.json'
        args=[(qa_list,i[0],i[1],i[2],i[3],i[4])for i in batch]
        mp_pool.starmap(worker,args)

        p.set_description_str(f"DataCount : {len(batch)}; {len(qa_list)}/{data_size}; {len(dataset['train'])}")
        if len(qa_list) > data_size:
            Update_file(list(qa_list),qadatapath)
            qa_list[:]=[]
            data_count+=1
            p.set_postfix_str(f"Write {qadatapath}")
    else:
        Update_file(list(qa_list),qadatapath)
    mp_pool.close()
    mp_pool.join()

if __name__=="__main__":
    main()
    # create_dataset()

