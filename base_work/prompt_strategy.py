class prompter:
    def __init__(self):
        self.setup_assis_prompt()

    def setup_assis_prompt(self):

        self.confidence_define_prompt="Note: The confidence indicates how likely you think your Answer is true and correct,from 0.00 (worst) to 1.00 (best)"

        self.similarity_prompt="Note: The similarity indicates how likely you think your Answer and document is semantic related,from 0.00 (worst) to 1.00 (best)"

        self.acc_prompt="Note: The accuracy indicates how likely you think your ground truth and answer have the same meaning, give 0 (wrong) or 1 (Correct)"

        self.responseformat="response format:\n'Answer':[ONLY Your final Answer here],\n'Confidence':[Your final Confidence here]"

    def setup_task(self,task):
        if task:
            if task=="QA":
                self.answer_type="provide Answer to the question and confidence to the Answer"
                self.system_prompt=f"This is a QA task, please {self.answer_type} in json."

            elif task=="Long_QA":

                self.answer_type="provide very long Answer with more details to the question and confidence to the Answer"
                self.system_prompt=f"This is a Long form generation QA task, {self.answer_type} in json."

            elif task=="similarity":

                self.system_prompt=f"This is a similarity compare task, please compare the semantic similarity and provide the score in json."

            elif task=="self_polish":

                self.system_prompt=f"Rewrite new versions of the original question to be more understandable and easy to answer. Don't omit any useful information. and please maintain their original meaning when polysemous words appear in json."

            elif task=="RaR":

                self.answer_type='answer the question'
                self.system_prompt=f"{self.answer_type} in json"

            elif task=='acc':

                self.system_prompt=f"This is a accuracy task, please judge if ground truth and answer have exactly same semantic meaning and provide the accuracy in json"

            elif task=="pure":

                self.answer_type='answer the question'
                self.system_prompt=f"{self.answer_type} in json"
        else:
            raise ValueError("task Not Recognized")

    def get_prompt(self,query:list,document:list,stretagy:str):

        # if with_rag:
        #     # logger.info(f"{stretagy} activate knwledge {with_rag}")
        #     doc_str=" ".join(document)
        #     self.rag_inject=f"base on the given Knowledge"
        #     self.document=f"Knwoledge : {doc_str},\n"
        if stretagy=="vanilla":
            return self.vanilla_prompt(query,document)

        elif stretagy=="cot":
            return self.chain_of_thought(query,document)

        elif stretagy=="multi_step":
            return self.multi_step(query,document)

        elif stretagy=="topk":
            return self.topk_prompt(query,document)

        elif stretagy=="similarity":
            return self.document_answer_similarity(query,document)

        elif stretagy=="self_polish":
            return self.self_polish_prompt(query)

        elif stretagy=="RaR":
            return self.RaR_prompt(query)

        elif stretagy=="acc":
            return self.answer_acc(query)


    def document_answer_similarity(self,answer:list,document:list)-> dict:
        # logger.info(f"{answer} {type(document)}")
        document_str="\n".join(document)
        answer="".join(answer)
        Instruction=f"Give the result of the accuracy between the Answer and the document"
        similarty_froamt="{accuracy:[Your final accuracy here]}"
        input_text=f"groudtruth:{document_str},\nAnswer:{answer},\n\nresponse format :{similarty_froamt}\n"

        return {"system_prompt":self.system_prompt,'Instruction':Instruction,"Question":"",'input_text':input_text,"assit_prompt":self.similarity_prompt}


    def answer_acc(self,query):
        if len(query) !=2:
            print("Query List size should be 2")
            exit()

        Instruction=f"Compare the semantic similarity between given groudtruth and Answer"
        accuracy_format="{accuracy:[Your final accuracy here]}"
        input_text=f"\nOnly give 0(wrong) or 1(correct) according to response format in json, don't give me any other words.\n\ngroudtruth:{query[0]},\nAnswer:{query[1]},\n\nresponse format :{accuracy_format}\n"

        return {"system_prompt":self.system_prompt,'Instruction':Instruction,"Question":"",'input_text':input_text,"assit_prompt":self.acc_prompt}

    def topk_prompt(self,question:list,document:list)-> dict:
        question=question.pop()

        Instruction=f"Now, Read the Question and {self.answer_type}\n"

        vanilla_prompt=f'''\nOnly give me one Answer and Confidence according to response format in json, don't give me any other words.\n\n{self.responseformat}'''

        return {"system_prompt":self.system_prompt,'Instruction':Instruction,"Question":f"Question : {question}",'input_text':vanilla_prompt,"assit_prompt":self.confidence_define_prompt}



    def vanilla_prompt(self,question:list,document:list)-> dict:
        question=question.pop()

        Instruction=f"Now, Read the Question and {self.answer_type}\n"

        vanilla_prompt=f'''\nOnly give me one Answer and Confidence according to response format in json, don't give me any other words.\n\n{self.responseformat}'''

        return {"system_prompt":self.system_prompt,'Instruction':Instruction,"Question":f"Question : {question}",'input_text':vanilla_prompt,"assit_prompt":self.confidence_define_prompt}

    def chain_of_thought(self,question:list,document:list)-> dict:
        question=question.pop()

        self.cotresponseformat="response format:\nAnswer:[ONLY Your final Answer here],\nConfidence:[Your final Confidence here]\Explanation:[Your Explanation here]"

        Instruction=f"Now, Read the Question and {self.answer_type}. Let's think step by step and give the Explanation to the Answer\n"

        cot_prompt = f'''\nOnly give me one Answer, Confidence and Explanation according to response format in json, don't give me any other words.\n\n{self.cotresponseformat}'''

        return {"system_prompt":self.system_prompt,'Instruction':Instruction,"Question":f"Question : {question}",'input_text':cot_prompt,"assit_prompt":self.confidence_define_prompt}

    def multi_step(self,question:list,document:list)->dict:
        question=question.pop()

        self.multistepresponseformat="response format:\nAnswer:[ONLY Your final Answer here],\nConfidence:[Your final Confidence here]\Step_result:[Your Explanation here]"

        step_prompt=f"Step 1: [Your reasoning]... Step k : [Your reasoning]"

        Instruction=f"Read the question, break down the problem into K steps, think step by step, give your confidence in each step, and then {self.answer_type}\n"

        multi_step_prompt = f'''\nOnly give me one reply according to response format in json, don't give me any other words.\n\n{step_prompt}\n{self.multistepresponseformat}'''

        return {"system_prompt":self.system_prompt,'Instruction':Instruction,"Question":f"Question : {question}",'input_text':multi_step_prompt,"assit_prompt":self.confidence_define_prompt}

    def self_polish_prompt(self,question:list)->dict:
        question=question.pop()

        polish_format="response format:\nNew_Question: [Your New Question here]"

        rewrite_prompt=f"\nOnly give me one New_Question according to response format in json, don't give me any other words.\n{polish_format}"

        return {"system_prompt":self.system_prompt,'Instruction':"","Question":f"Question : {question}",'input_text':rewrite_prompt,"assit_prompt":""}

    def RaR_prompt(self,question:list)->dict:
        question=question.pop()

        rar_responsformat="response format:\nExpanded_Question: [Your Expanded Question here]\nAnswer: [Your Answer here]\nConfidence: [Your Confidence here]"
        rar_prompt=f"\nRephrase and expand the question, and respond Answer and Confidence\nOnly give me one Answer and Confidence according to response format in json, don't give me any other words.\n\n{rar_responsformat}\n"

        return {"system_prompt":self.system_prompt,'Instruction':"","Question":f"Question : {question}",'input_text':rar_prompt,"assit_prompt":self.confidence_define_prompt}




