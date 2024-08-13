# CAPR: Confidence-Aware Prompt Refinement for Dehallucination

## Structure
```
./
├── api_key.yml
├── base_work
    ├──response_result
    └──baseline_result
├── CAPR
    ├──Agent_weight
    └──response_result
├── PACE
    └──response_result
├── pre_experiment.ipynb
└── README.md
```
## Preprocessing- Data Download
Please Download the .json File throught google drive link below and unzip to folder accccording to the structure above respsectively:
- [Google Link](https://drive.google.com/file/d/13blyu19dmVWNquZ7IzOKxP3GalaH6cwb/view?usp=sharing)
- Make Sure the api_key is provided in the api_key.yml format

## How to Run
### PACE
- Enter Folder **PACE**
    - Setup **datasets**, **strategies**, **api_model** for evaluation the result
        ```
            # 'natural_questions','din0s/asqa',"triviaQA",
            datasets = ["din0s/asqa"]
            strategies = ['vanilla','cot','multi_step']
            sim_models = 'Cos_sim'

            ## API model
            api_model = 'gpt-3.5-turbo-0125'
            # api_model = 'gpt-4-turbo'
            # api_model = 'claude-3-5-sonnet-20240620'
        ```
    - Run the bash command:
        ```
        python pipline.py
        ```
### CAPR

- Enter Folder **CAPR**
    - For **Inference**
        - Setup the different **Agent_addres** in **inference.py** for different setting:
            - For ASQA Dataset:
                - CAPR w/ PACE: **PPO_Agent_06122032_vanilla_f1_r12_withPACE_7_0.0012**
                - CAPR w/o PACE: **PPO_Agent_06122032_vanilla_f1_r11_withoutPACE_9_0.0009**
                - CAPR w/ ACC: **PPO_Agent_06122032_vanilla_f1_r12_OnlyReward_7_0.0007**
            - For TriviaQA Dataset:
                - CAPR w/ PACE: **PPO_Agent_06122032_vanilla_f1_r1_trivia_withPACE_7_0.0030**
        - Then Run the following bash command:
            ```
            python infernce.py
            ```
    - For **Traning**
        - Setup **Training_Config** for training detail:
            ```
            Training_Config={
                "dataset_path":f'response_result/20240601/triviaQA_gpt-3.5-turbo-0125_vanilla_QA.json', ## Training Data
                'deliminator':"06122032_vanilla_f1_r1_trivia_withPACE", ## Save_File deliminator
                'max_epoch': 8, ## Training Epoch
                'trian_batch_size':8, ## Training Batch Size
                'Batch_accumulate_size':32 ## Training Batch Accumulate Size min : 128, Max: 64
                            }
            ```
        - Then Run the following bash command:
            ```
            python TRL_training.py
            ```
### Baseline
- Contain **Vanilla** / **Self-Polish** / **Textgrad** / **Rephrease and Response(RaR)**
    - Modify Black-box model in Line 17-24
        ```
        #  gpt-3.5-turbo-0125, gpt-4-turbo
        # "claude-3-5-sonnet-20240620"

        # api_model='gpt-3.5-turbo-0125'
        # api_key=key['openai']['api_key']

        # api_model='gpt-4-turbo'
        # api_key=key['openai']['api_key']

        # api_model='claude-3-5-sonnet-20240620'
        # api_key=key['claude']['api_key']

        ```
    - Modify base_line.py for differnet baseline work in line 199-203
        ```
        baseline_list=["vanilla","self_polish","RaR"]
        for baseline in baseline_list:
            datapath=f'baseline_result/trivia_{api_model}_{baseline}_detail.json'
            main(baseline,datapath,'extract_answer')
            evaluate_result(datapath)
        ```
    - Run the code in terminal:
        ```
        python base_line.py
        ```

