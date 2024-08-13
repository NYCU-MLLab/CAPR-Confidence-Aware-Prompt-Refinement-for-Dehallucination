import os,glob,json
from util import load_checkpoint,Update_file

# dd=glob.glob("response_result/20240517/*.json")

# for datapath in dd:
#     with open(datapath,'r') as f:
#         data=f.read()

#     data=data.replace('confidence',"Confidence")
#     data=data.replace('Ground_truth',"Ground_Truth")
#     data=data.replace('ground truth',"Ground_Truth")
#     data=data.replace('Similarity_res',"Doc_Ans_simi")

#     with open(datapath,'w') as f:
#         f.write(data)

dd1=glob.glob("response_result/20240517/*f1.json")
dd2=glob.glob("response_result/20240517/*bertscore.json")
dd3=glob.glob("response_result/20240517/*_gpt-3.5-turbo-0125.json")
# print(dd3)

for datapath in dd1+dd2+dd3:
    acc_data=load_checkpoint(datapath)
    for idx,i in enumerate(acc_data):
        if None in i['Doc_Ans_simi']:
            print(i)
            # for idxj,jj in enumerate(i['Doc_Ans_simi']):
            #     if jj is None:
            #         acc_data[idx]['Doc_Ans_simi'][idxj]=0
    # Update_file(acc_data,datapath)
