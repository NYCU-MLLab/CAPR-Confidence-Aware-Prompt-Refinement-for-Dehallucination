
import os,json
from util import *
import glob
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve,precision_recall_curve
from sklearn.metrics import roc_curve, auc
from netcal.metrics import ECE
from show_graph import show_bar,load_eval_data

class SimpleStatsCache:
    def __init__(self, confids, correct):
        self.confids = np.array(confids)
        self.correct = np.array(correct)

    @property
    def rc_curve_stats(self):
        coverages = []
        risks = []

        n_residuals = len(self.residuals)
        idx_sorted = np.argsort(self.confids)

        coverage = n_residuals
        error_sum = sum(self.residuals[idx_sorted])

        coverages.append(coverage / n_residuals)
        risks.append(error_sum / n_residuals)

        weights = []

        tmp_weight = 0
        for i in range(0, len(idx_sorted) - 1):
            coverage = coverage - 1
            error_sum = error_sum - self.residuals[idx_sorted[i]]
            selective_risk = error_sum / (n_residuals - 1 - i)
            tmp_weight += 1
            if i == 0 or self.confids[idx_sorted[i]] != self.confids[idx_sorted[i - 1]]:
                coverages.append(coverage / n_residuals)
                risks.append(selective_risk)
                weights.append(tmp_weight / n_residuals)
                tmp_weight = 0

        # add a well-defined final point to the RC-curve.
        if tmp_weight > 0:
            coverages.append(0)
            risks.append(risks[-1])
            weights.append(tmp_weight / n_residuals)

        return coverages, risks, weights

    @property
    def residuals(self):
        return 1 - self.correct


def area_under_risk_coverage_score(confids, correct):
    stats_cache = SimpleStatsCache(confids, correct)
    _, risks, weights = stats_cache.rc_curve_stats
    AURC_DISPLAY_SCALE = 1000
    return sum([(risks[i] + risks[i + 1]) * 0.5 * weights[i] for i in range(len(weights))])* AURC_DISPLAY_SCALE


def compute_conf_metrics(y_acc, y_confs,Stretagy,acc_model,data_name):


    for idx,(acc,conf) in enumerate(zip(y_acc,y_confs)):
        if float(acc)==1.0:
            y_confs[idx]=1.0
        elif float(acc)==0.0:
            y_confs[idx]=0.0

    result_matrics = {}
    # ACC
    # accuracy = sum(y_true) / len(y_true)
    accuracy = np.mean(y_acc)
    print("accuracy: ", accuracy)
    result_matrics['acc'] = accuracy

    ## change to binary
    accuracy_bound={
        "f1" : 0.8,
        "bertscore" : 1.0,
        "rougeL" : 0.3
    }
    ##
    if acc_model in accuracy_bound:
        y_true=np.where(y_acc < accuracy_bound[acc_model],0,1)
    else:
        y_true=y_acc
    # use np to test if y_confs are all in [0, 1]

    assert all([x >= 0 and x <= 1 for x in y_confs]), y_confs ## makesure conf >0
    # y_confs, y_true = np.array(y_confs), np.array(y_true)

    # ROC
    Cal_roc_curve(y_true,y_confs,data_name)

    # AUCROC
    # roc_auc = roc_auc_score(y_true, y_confs)
    fpr1, tpr1, thresholds1 = roc_curve(y_true, y_confs)
    roc_auc=auc(fpr1, tpr1)
    print("ROC AUC score:", roc_auc)
    result_matrics['auroc'] = roc_auc

    # AUPRC-Positive
    auprc = average_precision_score(y_true, y_confs)
    print("AUC PRC Positive score:", auprc)
    result_matrics['auprc_p'] = auprc

    # AUPRC-Negative
    auprc = average_precision_score(1- y_true, 1 - y_confs)
    print("AUC PRC Negative score:", auprc)
    result_matrics['auprc_n'] = auprc

    # AURC from https://github.com/IML-DKFZ/fd-shifts/tree/main
    aurc = area_under_risk_coverage_score(y_true,y_confs)
    result_matrics['aurc'] = aurc
    print("AURC score:", aurc)

    # ECE
    n_bins = 10
    # diagram = ReliabilityDiagram(n_bins)
    ece = ECE(n_bins)
    ece_score = ece.measure(np.array(y_confs), np.array(y_true))
    print("ECE:", ece_score)
    result_matrics['ece'] = ece_score

    ece_all=np.mean(np.abs(np.array(y_acc)-np.array(y_confs)))
    print("ECE elements:", ece_all)
    result_matrics['ece_element'] = ece_all

    return result_matrics

def Cal_roc_curve(y_true,y_scores,data_name):
    os.makedirs("picture/roc_curve",exist_ok=True)
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Compute AUROC
    auc = roc_auc_score(y_true, y_scores)
    print(f"AUROC: {auc:.4f}")

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(f"picture/roc_curve/{data_name}_roc_curve.png")
    # plt.show()
    plt.clf()


def evaluate_score(api_model,dataset_path,Stretagy,sim_model,acc_model,activation_time,task,acc_datapath,lambda_value,shuffle):
    shuffle_str="shuffle" if shuffle else "No_shuffle"
    eval_datapath=f"response_result/Evaluate_Result_{activation_time}_{shuffle_str}.json"
    print(f"Path : {eval_datapath}")
    ## Load Checkpoint
    if os.path.isfile(acc_datapath):
        acc_data=load_checkpoint(acc_datapath)
    else:
        logger.info(f"{acc_datapath} not Exists")
        exit()

    if os.path.isfile(eval_datapath):
        evaluate_result=load_checkpoint(eval_datapath)
    else:
        evaluate_result=[]
    ##################
    ## init
    data_limit=4000

    if sim_model=="gpt-3.5-turbo-0125":
        max_simi=[max(map(float,i['Doc_Ans_simi'])) for i in acc_data]
        sim_array=np.array(max_simi)[:data_limit]
        acc_array=np.array([float(v['Accuracy']) for v in acc_data])[:data_limit]
        conf_array=np.array([float(v['Confidence']) for v in acc_data])[:data_limit]
        pace_conf_array=np.add(lambda_value*conf_array,(1-lambda_value)*sim_array)[:data_limit]

    elif sim_model=="snli":
        mean_simi=[sum(map(float,i['Doc_Ans_simi']))/len(i['Doc_Ans_simi']) for i in acc_data]
        max_simi=[max(map(float,i['Doc_Ans_simi'])) for i in acc_data]
        sim_array=np.array(max_simi)[:data_limit]
        acc_array=np.array([float(v['Accuracy']) for v in acc_data])[:data_limit]
        conf_array=np.array([float(v['Confidence']) for v in acc_data])[:data_limit]
        pace_conf_array=np.add(lambda_value*conf_array,(1-lambda_value)*sim_array)[:data_limit]

    elif sim_model=="Cos_sim":
        max_simi=[max(map(float,i['Doc_Ans_simi'])) for i in acc_data]
        mean_simi=[sum(map(float,i['Doc_Ans_simi']))/len(i['Doc_Ans_simi']) for i in acc_data]
        sim_array=np.array(max_simi)[:data_limit]
        acc_array=np.array([float(v['Accuracy']) for v in acc_data])[:data_limit]
        conf_array=np.array([float(v['Confidence']) for v in acc_data])[:data_limit]
        pace_conf_array=np.add(lambda_value*conf_array,(1-lambda_value)*sim_array)[:data_limit]

    # print(conf_array)
    # print(pace_conf_array)
    dataset_path=dataset_path.replace('/','_')
    ####################
    print("*"*100)
    print(f"{dataset_path} {Stretagy} {sim_model}")
    print("*"*50)
    print(f"Similarity Mean:{np.mean(sim_array)}")

    print("*"*50)
    print(f"Confidence Mean:{np.mean(conf_array)}")
    print("*"*50)
    print(f"PACE Confidence Mean: {np.mean(pace_conf_array)}")
    ## ECE and AUROC
    print("*"*50)
    print(f"{sim_model} {Stretagy} Without PACE")

    eval_result=compute_conf_metrics(acc_array,conf_array,Stretagy,acc_model,f"{dataset_path}_{Stretagy}_{sim_model}")
    print("*"*50)
    print(f"{sim_model} {Stretagy} With PACE")
    eval_pace_result=compute_conf_metrics(acc_array,pace_conf_array,Stretagy,acc_model,f"PACE_{dataset_path}_{Stretagy}_{sim_model}")
    print("*"*100)

    ## MacroCE

    ## TODO: MacroCE implementation

    ## Save
    New_evaluate_resul={
        'dataset':dataset_path,
        "Stratagy":Stretagy,
        'api_model':api_model,
        'sim_model':sim_model,
        'acc_model':acc_model,
        'Conf':list(conf_array),
        'Simi':list(sim_array),
        'Pace_Conf':list(pace_conf_array),
        'Accuracy':list(acc_array),
        'ece':eval_result['ece'],
        'ece_element':eval_result['ece_element'],
        'auroc':eval_result['auroc'],
        'auprc_p':eval_result['auprc_p'],
        'auprc_n':eval_result['auprc_n'],
        'aurc':eval_result['aurc'],
        'ece_pace':eval_pace_result['ece'],
        'ece_pace_element':eval_pace_result['ece_element'],
        'auroc_pace':eval_pace_result['auroc'],
        'auprc_p':eval_pace_result['auprc_p'],
        'auprc_n':eval_pace_result['auprc_n'],
        'aurc_pace':eval_pace_result['aurc'],
        "Evaluate_time":activation_time,
    }
    for idx,i in enumerate(evaluate_result):
        if New_evaluate_resul['dataset']==i['dataset'] and New_evaluate_resul['api_model']==i['api_model'] and New_evaluate_resul['Stratagy']==i['Stratagy'] and New_evaluate_resul['sim_model']==i['sim_model'] and New_evaluate_resul['acc_model']==i['acc_model']:
            evaluate_result[idx]=New_evaluate_resul
            break
    else:
        evaluate_result.append(New_evaluate_resul)

    Update_file(list(evaluate_result),eval_datapath)


def show(actime=20240601,API_model="",acc_model="",sim_model=""):
    with open(f"response_result/Evaluate_Result_{actime}_No_shuffle.json",'r') as f:
        data=json.load(f)
    path=f"Evaluate_show_{acc_model}_{actime}.json"
    with open(path,'w+') as f:
        f.writelines(f"|Dataset|Prompt Stretagy|API model|Similarity Model|ACC Model|Confidence Mean|Similarity Mean|Acc Mean|ECE|AUROC|auprc_p|auprc_n|ECE_PACE|AUROC_PACE|Evaluate Date|\n|:-|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|\n")
    for da in data:
        if da["api_model"]==API_model and da['acc_model']==acc_model and da['sim_model']==sim_model:
            print(f"Evaluate Result:")
            with open(path,'a+') as f:
                f.writelines(f"|")
                for k,v in da.items():
                    f.writelines(f"{v}|")

                    print(f"        {k} : {v}")
                else:
                    f.writelines(f"\n")

def count_down(path):
    if os.path.isfile(path):
        with open(path,'r') as f:
            return len(json.load(f))
    else:
        return 0

def Savefig(File_name,api_model,dataset,sim_models,acc_model_mapping,strategies):
    data_path=f"response_result/Evaluate_Result_{File_name}.json"
    ece_for_all,ece_pace_for_all=[],[]
    auroc_for_all,auroc_pace_for_all=[],[]
    aurc_for_all,aurc_pace_for_all=[],[]
    for j in strategies:
        ece,ece_pace=[],[]
        auroc,auroc_pac=[],[]
        aurc,aurc_pac=[],[]
        for i in dataset:
            data_eval=load_eval_data(i,acc_model_mapping[i],data_path,j,sim_models,api_model,File_name)
            if data_eval is not None:
                ece.append(data_eval['ece'])
                ece_pace.append(data_eval['ece_pace'])
                auroc.append(data_eval['auroc'])
                auroc_pac.append(data_eval['auroc_pace'])
                aurc.append(data_eval['aurc'])
                aurc_pac.append(data_eval['aurc_pace'])

        ece_for_all.append(ece)
        ece_pace_for_all.append(ece_pace)
        auroc_for_all.append(auroc)
        auroc_pace_for_all.append(auroc_pac)
        aurc_for_all.append(aurc)
        aurc_pace_for_all.append(aurc_pac)

    if ece_for_all and auroc_for_all and aurc_for_all:
        ## vector_list,label,catagoriers_list,x_label,y_label,title
        ece_result=[[ece_for_all,strategies,dataset,'Category','ece',f"ECE",(0,1)],[ece_pace_for_all,strategies,dataset,'Category','ece',f"ECE_pace",(0,1)]]
        auroc_result=[[auroc_for_all,strategies,dataset,'Category','auroc',f"AUROC",(0,1)],[auroc_pace_for_all,strategies,dataset,'Category','auroc',f"AUROC_pace",(0,1)]]
        aurc_result=[[aurc_for_all,strategies,dataset,'Category','aurc',f"AURC",(0,1000)],[aurc_pace_for_all,strategies,dataset,'Category','aurc',f"AURC_pace",(0,1000)]]
        print("*"*100)
        print(f"{sim_models}")
        print(ece_result)
        print(auroc_result)
        print(aurc_result)
        print("*"*100)
        show_bar(ece_result,f"ECE_RESULT_{api_model}_{File_name}")
        show_bar(auroc_result,f"AUROC_result_{api_model}_{File_name}")
        show_bar(aurc_result,f"AURC_result_{api_model}_{File_name}")

if __name__=="__main__":
    activate_time="20240601"
    # sim_model=["Cos_sim"]
    # acc_model="f1"
    # dataset=["din0s_asqa"]
    # stretagy=["vanilla","cot","multi_step"]
    # api_model="gpt-3.5-turbo-0125"
    # task={
    #     "natural_questions":'QA',
    #     'din0s_asqa':'Long_QA'
    #     }
    # for k in stretagy:
    #     for j in dataset:
    #         for i in sim_model:
    #             acc_path=f"response_result/{activate_time}/{j}_{api_model}_{k}_{task[j]}_{i}_No_Shuffle_{acc_model}.json"
    #             if os.path.isfile(acc_path):
    #                 print(acc_path)
    #                 config={
    #                     "api_model":f"{api_model}",
    #                     "dataset_path":f"{j}",
    #                     "sim_model":f"{i}",
    #                     "acc_model":f"{acc_model}",
    #                     "activation_time":f"{activate_time}",
    #                     "task":f"{task[j]}",
    #                     'Stretagy':f"{k}",
    #                     "acc_datapath":acc_path,
    #                     "lambda_value":0.5,
    #                     "shuffle":False
    #                 }
    #                 evaluate_score(**config)
    #             else:
    #                 print(f"{acc_path} not exist")
    ## Show Result and Cost
    # show(20240521,"gpt-3.5-turbo-0125","bertscore","snli")
    # show(20240520,"gpt-3.5-turbo-0125","f1","gpt-3.5-turbo-0125")
    # show(20240521,"gpt-3.5-turbo-0125","wer","gpt-3.5-turbo-0125")
    show(20240601,"gpt-3.5-turbo-0125","rougeL","gpt-3.5-turbo-0125")
    # Savefig(f"{activate_time}_No_shuffle")
    # spent=Get_Cost()
    # print(f"Sprent {spent} USD {spent*30} NTD\n")


