import json,os
import matplotlib.pyplot as plt


def Moving_average(data_folder):
    data_size=20
    ratio=0.99
    for k in ['reward','Accuracy','ECE']:
        data_path=f'{data_folder}/{k}.json'
        if os.path.isfile(data_path):
            with open(data_path,'r') as f:
                data=json.load(f)
            movin_avg=sum(data[:data_size])/data_size
            k1=[]
            for i in data:
                movin_avg=ratio*movin_avg+(1-ratio)*i
                k1.append(movin_avg)

            plt.plot(range(len(k1)),k1,label=k,marker='')
            plt.legend()
            plt.savefig(f"{data_path.replace(".json","mvavg.png")}")
            plt.show()
            plt.clf()

if __name__=="__main__":
    Moving_average("CAPR/PPO_State_06122032_vanilla_f1_r1_trivia_withPACE")
