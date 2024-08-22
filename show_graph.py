# %%
import os,json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from util import *
import matplotlib.pyplot as plt
import numpy as np
import json,random,os
from sklearn.calibration import calibration_curve

# datapath="response_result/20240517/din0s_asqa_gpt-3.5-turbo-0125_cot_Long_QA_gpt-3.5-turbo-0125_bertscore.json"
activation_time=datetime.now().strftime("%Y_%m_%d")
checkpoint_time=f'2024_04_30'
def get_datapath(dataset,api_model,activate_time,acc_model,sim_model,prompt_strategy,isshuffle_str)-> list:
    task=["QA","Long_QA"]
    datapath=[]
    for i in prompt_strategy:
        for t in task:
            path=f"response_result/{activate_time}/{dataset}_{api_model}_{i}_{t}_{sim_model}_{acc_model}_{isshuffle_str}.json"
            print(path)
            if os.path.isfile(path):
                datapath.append(path)

    return datapath

def mean(data):
    return sum(data)/len(data)

def Load_data(datapath):
    with open(datapath,'r') as f:
        data=json.load(f)

    conf=[i['Confidence'] for i in data]
    ## MAX
    simi=[max(map(float,i['Doc_Ans_simi'])) for i in data]
    ## Mean
    mean_simi=[sum(map(float,i['Doc_Ans_simi']))/len(i['Doc_Ans_simi']) for i in data]
    acc=[i['Accuracy'] for i in data]
    assert len(simi)==len(conf)
    return [conf,mean_simi,acc]


# color=['lightblue','lightred',"lightgreen","yellow",'pink','lightbrown']
def Get_histogram(datalist,dataset,title,stretagy):
# Generate sample data
    # data1 = np.random.normal(0, 1, 1000)
    plt.figure(figsize=(4, 3))
    colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
    # assert len(lable)==len(datalist)
    # Plot histograms
    for idx,i in enumerate(datalist):
        plt.hist(i, bins=100, alpha=0.7, label=stretagy[idx],color=colors[idx% len(colors)])
    # Adding labels and title
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f"{dataset}_{title}")
    plt.legend(loc='upper right')
    plt.xlim([0, 1])
    # plt.ylim([0, 600])
    # Show the plot
    plt.savefig(f"picture/histogram/{dataset}_{title}.png")
    # plt.show()
    plt.clf

def show_histogram_graph(vector,title,File_name,stretagy="",sim="",datafile_name="",label=[]):
    os.makedirs(f"PACE/picture/histogram/{File_name}",exist_ok=True)
    # Plot histogram
    plt.figure(figsize=(4, 3))
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
    plt.savefig(f"PACE/picture/histogram/{File_name}/{datafile_name}.png")
    # plt.show()
    plt.clf

def show_plot_graph(vector,File_name,datafile_name="",label=""):
    os.makedirs(f"PACE/picture/histogram/{File_name}",exist_ok=True)
    # Plot histogram
    plt.figure(figsize=(6, 3),facecolor='none')
    colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
    random.shuffle(colors)
    ## len(vector)//5
    counts, bin_edges = np.histogram(vector, bins=10)
    plt.gca().set_facecolor(None)
    # 計算每個柱的中心點
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # plt.hist(vector, bins=30, alpha=0.3, color='gray', edgecolor='black')
    plt.plot(bin_centers,counts, marker='',linestyle='-', color=colors[random.randint(0,len(colors)-1)],label=label,linewidth=3)
    # Add a grid
    # plt.grid(True)
    plt.xlim(0,1)
    plt.legend(loc='upper left', fontsize=10)
    # Show plot
    plt.savefig(f"PACE/picture/histogram/{File_name}/{datafile_name}.png",transparent=True)
    # plt.show()
    plt.clf


def calibration_curve_fucntion_single(accuracy1,Confidence1,accuracy2,Confidence2,n_bins=10,File_name="",datafile_name="",labely="",labelx="",label_legend="",color_index=0):
    os.makedirs(f"PACE/picture/histogram/{File_name}",exist_ok=True)
    # Calculate the calibration curve
    accuracy1=np.where(np.array(accuracy1)<0.3,0,1)
    accuracy2=np.where(np.array(accuracy2)<0.3,0,1)
    prob_true1, prob_pred1 = calibration_curve(accuracy1,Confidence1, n_bins=n_bins)
    prob_true2, prob_pred2 = calibration_curve(accuracy2,Confidence2, n_bins=n_bins)

    # Plot the reliability diagram
    plt.figure(figsize=(4, 4),facecolor='none',dpi=300)
    colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
    print(colors)
    # random.shuffle(colors)

    random_color=random.randint(0,len(colors)-1)

    plt.plot(prob_pred1,prob_true1 , marker='^', label=f'{label_legend} calibration curve',color=colors[color_index%len(colors)],linewidth=2,markersize=7)
    # plt.plot(prob_pred2,prob_true2 , marker='o', label=f'{label_legend} w/ PACE',color=colors[(color_index+1)%len(colors)],linewidth=2,markersize=8)
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated',color='black',linewidth=2)
    plt.gca().set_facecolor(None)
    plt.xlabel(f'{labelx}',fontsize=7)
    plt.ylabel(f'Accuracy',fontsize=7)
    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)

    plt.legend(fontsize=10)
    plt.grid()
    # plt.show()
    plt.savefig(f"PACE/picture/histogram/{File_name}/{datafile_name}_single.png",transparent=True)
    plt.clf


def calibration_curve_fucntion(accuracy1,Confidence1,accuracy2,Confidence2,n_bins=10,File_name="",datafile_name="",labely="",labelx="",label_legend="",color_index=0):
    os.makedirs(f"PACE/picture/histogram/{File_name}",exist_ok=True)
    # Calculate the calibration curve
    accuracy1=np.where(np.array(accuracy1)<0.3,0,1)
    accuracy2=np.where(np.array(accuracy2)<0.3,0,1)
    prob_true1, prob_pred1 = calibration_curve(accuracy1,Confidence1, n_bins=n_bins)
    prob_true2, prob_pred2 = calibration_curve(accuracy2,Confidence2, n_bins=n_bins)

    # Plot the reliability diagram
    plt.figure(figsize=(4, 4),facecolor='none',dpi=300)
    colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
    print(colors)
    # random.shuffle(colors)

    random_color=random.randint(0,len(colors)-1)

    plt.plot(prob_pred1,prob_true1 , marker='^', label=f'{label_legend} w/o PACE',color=colors[color_index%len(colors)],linewidth=2,markersize=7)
    plt.plot(prob_pred2,prob_true2 , marker='o', label=f'{label_legend} w/ PACE',color=colors[(color_index+1)%len(colors)],linewidth=2,markersize=7)
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated',color='black',linewidth=2)
    plt.gca().set_facecolor(None)
    plt.xlabel(f'{labelx}',fontsize=7)
    plt.ylabel(f'Accuracy',fontsize=7)
    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)

    plt.legend(fontsize=10)
    plt.grid()
    # plt.show()
    plt.savefig(f"PACE/picture/histogram/{File_name}/{datafile_name}.png",transparent=True)
    plt.clf


def clean_data(accuracy,conf):
    clean_acc,clean_conf=[],[]
    for ac,cof in zip(accuracy,conf):
        if cof < 0.5:
            print(ac,cof)
        else:
            clean_acc.append(ac)
            clean_conf.append(cof)
    return np.array(clean_acc),np.array(clean_conf)

def Calibration_result(activate_time,shuffle):
    isshuffle_str="shuffle" if shuffle else "No_shuffle"
    datapaht=f"./PACE/response_result/Evaluate_Result_{activate_time}_{isshuffle_str}.json"
    with open(datapaht,'r') as f:
        data=json.load(f)

    os.makedirs("PACE/picture/histogram",exist_ok=True)
    simi_models=["Cos_sim"]
    datasets=["din0s/asqa","triviaQA"]
    # datasets=["triviaQA"]
    api_model='gpt-3.5-turbo-0125'

    acc_mapping={"din0s/asqa":"rougeL","triviaQA":"f1"}
    label_mapping={"din0s/asqa":"ASQA","triviaQA":"TriviaQA"}

    # stretagy=["vanilla",'cot',"multi_step"]
    stretagy=["vanilla"]
    color_index=2
    for sim in simi_models:
        for idx,dataset in enumerate(datasets):
            color_index+=2
            for dd in data:
                for k in stretagy:
                    # conf_list,Final_conf_list,simi_list,acc_list=[],[],[],[]
                    if dd['dataset']==dataset and dd['sim_model']==sim and dd['Stratagy']==k and dd['acc_model']==acc_mapping[dataset] and api_model==dd['api_model']:
                        dataset_path=dataset.replace("/","_")
                        print(f"Load Sucess {dataset} {k} {sim}")

                        acc,pace_conf=clean_data(dd['Accuracy'],dd['Pace_Conf'])
                        acc,conf=clean_data(dd['Accuracy'],dd['Conf'])

                        ### Show plot Figure
                        show_plot_graph(dd['Conf'],File_name=f"{activate_time}_{isshuffle_str}",datafile_name=f"{dataset_path}_{isshuffle_str}_{sim}_{k}_Confidence",label=f"Confidence")

                        show_plot_graph(dd['Simi'],File_name=f"{activate_time}_{isshuffle_str}",datafile_name=f"{dataset_path}_{isshuffle_str}_{sim}_{k}_Similarity",label=f"Similarity")

                        show_plot_graph(dd['Pace_Conf'],File_name=f"{activate_time}_{isshuffle_str}",datafile_name=f"{dataset_path}_{isshuffle_str}_{sim}_{k}_PACE_Confidence",label=f"PACE Confidence")

                        show_plot_graph(dd['Accuracy'],File_name=f"{activate_time}_{isshuffle_str}",datafile_name=f"{dataset_path}_{isshuffle_str}_{sim}_{k}_Accuracy",label=f"Accuracy")

                        ### Show Accuracy and Confidence figure
                        calibration_curve_fucntion(acc,conf,acc,pace_conf,40,File_name=f"{activate_time}_{isshuffle_str}",datafile_name=f"{dataset_path}_calibration_curve",labely=f"{acc_mapping[dataset]}",labelx="Confidence",label_legend=f"{label_mapping[dataset]}",color_index=color_index)
                        print("*"*100)
                        calibration_curve_fucntion_single(acc,conf,acc,pace_conf,40,File_name=f"{activate_time}_{isshuffle_str}",datafile_name=f"{dataset_path}_calibration_curve",labely=f"{acc_mapping[dataset]}",labelx="Confidence",label_legend=f"{label_mapping[dataset]}",color_index=color_index)



def show_histogram_graph(vector,dim_x_y,stretagy=""):
    # Plot histogram
    plt.hist(vector, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
    # Add a title and labels
    plt.title(f'{stretagy} Score')
    plt.xlabel('Confidence Value')
    plt.ylabel('Density')
    plt.ylim(dim_x_y[1])
    plt.xlim(dim_x_y[0])
    # Add a grid
    plt.grid(True)

    # Show plot
    plt.show()

def plt_result(vector,stretagy="",color='blue'):
    sample_indices = range(0, len(vector))
    x_smooth = np.linspace(min(sample_indices), max(sample_indices), len(vector))
    y_smooth = np.interp(x_smooth, sample_indices, vector)
    # plt.plot(x_smooth, y_smooth,marker="o",linestyle='-')
    plt.scatter(sample_indices, vector, color=color, label='Sample Data',s=10)
    # plt.plot(vector,marker='o',linestyle='-')
        # Add a title and labels
    plt.title(f'{stretagy} Score')
    plt.xlabel('Confidence Value')
    plt.ylabel('Density')
    plt.ylim(0.0,1.0)
    # Add a grid
    plt.grid(True)

    # Show plot
    plt.show()

def show_gaussian(mu,sigma):
    # Generate data points for the Gaussian distribution
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    y = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    # Plot the Gaussian distribution
    plt.plot(x, y, color='blue', label='Gaussian Distribution')

    # Add a title and labels
    plt.title('Gaussian Distribution')
    plt.xlabel('Value')
    plt.ylabel('Density')

    # Add a legend
    plt.legend()

    # Show plot
    plt.show()


def Get_histogram(datalist,dataset,title):
# Generate sample data
    # data1 = np.random.normal(0, 1, 1000)

    colors=plt.rcParams['axes.prop_cycle'].by_key()['color']

    lable=["Vanilla","Cot","multi_step"]
    assert len(lable)==len(datalist)
    # Plot histograms
    for idx,i in enumerate(datalist):
        plt.hist(i, bins=50, alpha=0.7, label=lable[idx],color=colors[idx% len(colors)])


    # Adding labels and title
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f"{dataset}_{title}")
    plt.legend(loc='upper right')
    plt.xlim([0, 1])
    # plt.ylim([0, 600])
    # Show the plot
    plt.show()
    plt.savefig(f"histogram/{dataset}_{title}.png")
    plt.clf

def overall_conf(vec1,vec2,sample_size=500,lambda_x=0.5):
    return [lambda_x*i+(1-lambda_x)*j for i,j in zip(vec1[:sample_size],vec2[:sample_size])]


def load_eval_data(dataset_name="",acc_model_name="",datapath="",stretagy="",sim_model="",api_model="",activation_time=""):
    if os.path.isfile(datapath):
        with open(datapath,'r') as f:
            data=json.load(f)
        for i in data:
            if i["dataset"]==dataset_name  and i["acc_model"]==acc_model_name and i['Stratagy']==stretagy and i['sim_model']==sim_model and i['api_model']==api_model:
                print(f"{dataset_name},{acc_model_name},{stretagy},{sim_model},{api_model} Exists")
                return i
        else:
            print(f"{dataset_name},{acc_model_name},{stretagy},{sim_model},{api_model} Not Exists")
    else:
        print(f"{datapath} Do not Exists")
        return None
def show_two_subfigure(vec1, vec2, strategy:list,form='hist'):
    # Create a figure and subplots
    fig, axs = plt.subplots(1, 2,figsize=(10, 4))

    axs=setup_subfigure(vec1,'Vanilla Prompt',axs,[0],[(0.0,1.0),None],"Confidence",'Density','blue','hist')
    axs=setup_subfigure(vec1,'COT Prompt',axs,[1],[(0.0,1.0),None],"Confidence",'Density','blue','hist')

    # Adjust layout
    plt.tight_layout()
    # Add a grid
    plt.grid(True)
    # Show plot
    plt.show()

def show_four_subfigure( vec:list, strategy:list):
    # Create a figure and subplots
    fig, axs = plt.subplots(2, 2,figsize=(10, 8))

    axs=setup_subfigure(vec[0],strategy[0],axs,[(0,0)],[None,(0.0,1.0)],"Index",'Confidence','blue','scatter')
    axs=setup_subfigure(vec[1],strategy[1],axs,[(0,1)],[None,(0.0,1.0)],"Index",'Confidence','pink','scatter')
    axs=setup_subfigure(vec[2],strategy[2],axs,[(1,0)],[None,(0.0,1.0)],"Index",'Confidence','red','scatter')
    axs=setup_subfigure(vec[3],strategy[3],axs,[(1,1)],[None,(0.0,1.0)],"Index",'Confidence','green','scatter')

    # Adjust layout
    plt.tight_layout()
    # Add a grid
    plt.grid(True)
    # Show plot
    plt.show()

def setup_subfigure(vector,title,axs,idx:list,dim_x_y:list,x_label,y_label,color,form):

    index=idx.pop()
    if form=="hist":
        axs[index].hist(vector, bins=10, density=True, alpha=0.7, color=color, edgecolor='black')
    elif form=="scatter":
        axs[index].scatter(range(0, len(vector)), vector, color=color, label='Sample Data', s=10)

    axs[index].set_title(f'{title}')
    axs[index].set_xlabel(f'{x_label}')
    axs[index].set_ylabel(f'{y_label}')
    axs[index].set_ylim(dim_x_y[1])
    axs[index].set_xlim(dim_x_y[0])
    return axs

def show_bar(data_evaluation, png_name="bar_pic"):
    os.makedirs('picture', exist_ok=True)
    colors = ['darkred', 'darkblue', 'darkgreen', 'darkorange', 'pink', "grey"]
    bar_width = 0.2
    pic_number = len(data_evaluation)

    fig, axs = plt.subplots(1, pic_number, figsize=(8, 4))

    # Ensure axs is iterable in case there is only one subplot
    if pic_number == 1:
        axs = [axs]

    for i in range(pic_number):
        vector_list, label, categories_list, x_label, y_label, title, y_lims = data_evaluation[i]
        r_init = np.arange(len(vector_list[0]))

        for idx, vec in enumerate(vector_list):
            # Plotting the bars
            bars = axs[i].bar(r_init, vec, color=colors[idx % len(colors)], width=bar_width, edgecolor='grey', label=label[idx])

            for bar in bars:
                yval = bar.get_height()
                axs[i].text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.4f}', ha='center', va='bottom')

            r_init = [x + bar_width for x in r_init]

        # Add labels
        axs[i].set_xlabel(x_label, fontweight='bold')
        axs[i].set_ylabel(y_label, fontweight='bold')
        axs[i].set_xticks(np.arange(len(categories_list)) + bar_width * (len(vector_list) - 1) / 2)
        axs[i].set_xticklabels(categories_list)
        if y_lims and None not in y_lims:
            axs[i].set_ylim(y_lims)
            axs[i].set_yticks(np.arange(0, y_lims[1] + 0.1, y_lims[1] / 10))

        mean_value = np.mean([val for sublist in vector_list for val in sublist])
        axs[i].axhline(y=mean_value, color='grey', linestyle='--', linewidth=1)  # Add standard dashed line

        # Add title and legend
        # axs[i].set_title(title)
        axs[i].legend()

        # Save plot to file
    fig.savefig(f'picture/{png_name}.png')
    plt.clf()

def Get_data_dict(activate_time,shuffle):
    isshuffle_str="shuffle" if shuffle else "No_shuffle"
    datapaht=f"./PACE/response_result/Evaluate_Result_{activate_time}_{isshuffle_str}.json"
    with open(datapaht,'r') as f:
        data=json.load(f)

    os.makedirs("PACE/picture/histogram",exist_ok=True)
    simi_models=["Cos_sim"]
    datasets=["din0s/asqa","triviaQA"]
    # datasets=["triviaQA"]
    api_model='gpt-3.5-turbo-0125'

    acc_mapping={"din0s/asqa":"rougeL","triviaQA":"f1"}
    label_mapping={"din0s/asqa":"ASQA","triviaQA":"TriviaQA"}

    stretagy=["vanilla",'cot',"multi_step"]
    # stretagy=["vanilla"]
    color_index=2
    data_dict={}
    for sim in simi_models:
        for idx,dataset in enumerate(datasets):
            color_index+=2
            for dd in data:
                for k in stretagy:
                    # conf_list,Final_conf_list,simi_list,acc_list=[],[],[],[]
                    if dd['dataset']==dataset and dd['sim_model']==sim and dd['Stratagy']==k and dd['acc_model']==acc_mapping[dataset] and api_model==dd['api_model']:
                        dataset_path=dataset.replace("/","_")
                        print(f"Load Sucess {dataset} {k} {sim}")
                        # acc,pace_conf=clean_data(dd['Accuracy'],dd['Pace_Conf'])
                        # acc,conf=clean_data(dd['Accuracy'],dd['Conf'])
                        print(dataset)
                        if dataset not in data_dict:
                            data_dict[dataset]={}
                        if k not in data_dict[dataset]:
                            data_dict[dataset][k]={}

                        data_dict[dataset][k]["ECE"]=dd['ece']
                        data_dict[dataset][k]["ECE_PACE"]=dd['ece_pace']
                        data_dict[dataset][k]["AUROC"]=dd['auroc']
                        data_dict[dataset][k]["AUROC_PACE"]=dd['auroc_pace']
    return data_dict



if __name__=="__main__":
    data_dict=Get_data_dict("20240601",False)
    breakpoint
    Calibration_result("20240601",False)



