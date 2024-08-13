import torch,yaml,os
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bert_model import BERTSequenceClassifier
from util import setup_logger
from qadataset_dataloader import eval_dataloader
logger=setup_logger("log/trian_snli_classifier.log")

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

label_dict={
        0:"entailment",
        1:"neutral",
        2:"contradiction",
    }

def snli_similarity(model,tokenizer,premise:str,hypothesis:str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Define a function for inference
    with torch.no_grad():
        model.eval()
        inputs=tokenizer(premise+hypothesis,truncation=True,padding=True,max_length=1024,return_tensors='pt')
        input_ids=inputs.input_ids.to(device)
        attention_mask=inputs.attention_mask.to(device)
        outputs = model(input_ids,attention_mask=attention_mask)
        probabilities = torch.max(outputs).item()
        predicted_label = torch.argmax(outputs, dim=-1).item()
        return predicted_label,probabilities,outputs.cpu().numpy()

def Load_snli_model(prestraned_weight_path=f"bert_model_weight/bert_snli_classifier_eval_M3_0.89.pth"):
    model = BERTSequenceClassifier(num_classes=3)
    if os.path.isfile(prestraned_weight_path):
        pre_data=torch.load(prestraned_weight_path)
        model.load_state_dict(pre_data['model_state_dict'])
        # logger.info(f"Load From {prestraned_weight_path}")

    # Load BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['trian_bert_config']['tokenizer_name'])

    return model,tokenizer

def chunk_document(batch_data:str):
    TOKEN_LIMIT = 512
    # 使用tokenizer将文章转换为tokens
    tokenizer = AutoTokenizer.from_pretrained(config['trian_bert_config']['tokenizer_name'])
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokens = tokenizer(batch_data, return_tensors='pt',padding=True).input_ids
    # ['input_ids']
    # 将tokens切分成固定大小的块

    # chunks = [tokens[i*TOKEN_LIMIT:(i+1 )*TOKEN_LIMIT] for i in range(0, len(tokens), TOKEN_LIMIT)]

    # 将每个块转换回字符串
    chunks_text = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in tokens]

    # # 打印每个块
    # for i, chunk in enumerate(chunks_text):
    #     print(f"Chunk {i+1}:\n{chunk}\n")
    return chunks_text


if __name__=="__main__":
    tokenizer = AutoTokenizer.from_pretrained(config['trian_bert_config']['tokenizer_name'])
    simi_dataloader=eval_dataloader('response_result/20240520/natural_questions_gpt-3.5-turbo-0125_vanilla_QA.json',batch_size=1,purpose='compare').loader

    premise="Today is a good day"
    hypothesis="Today is Friday"
    m,t=Load_snli_model()
    predicted_label,probabilities,outputs=snli_similarity(m,t,premise,hypothesis)
    print(predicted_label)
    print(probabilities)
    print(outputs)
