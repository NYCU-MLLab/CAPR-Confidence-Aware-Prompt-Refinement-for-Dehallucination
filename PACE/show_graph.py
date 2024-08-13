# %%
import os,json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from util import *
activation_time=datetime.now().strftime("%Y_%m_%d")
checkpoint_time=f'2024_04_30'


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


if __name__=="__main__":
    # Example usage:
    data_evaluation = [
        (
            [[0.1], [0.2], [0.3]],
            ['Label1'],
            ['Category1', 'Category2', 'Category3'],
            'X Axis Label 1',
            'Y Axis Label 1',
            'Title1',
            [None,None],

        ),
    ]
    show_bar(data_evaluation)



