import os
import gradio as gr
import random
import torch
# import cv2
import re
import uuid
from PIL import Image, ImageDraw, ImageOps
import math
import numpy as np
import argparse
import inspect

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

# from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline, StableDiffusionInstructPix2PixPipeline
# from diffusers import EulerAncestralDiscreteScheduler
# from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
# from controlnet_aux import OpenposeDetector, MLSDdetector, HEDdetector

from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationSummaryMemory, ConversationBufferWindowMemory
from langchain.llms.openai import OpenAI
from langchain import PromptTemplate, HuggingFaceHub, LLMChain, HuggingFacePipeline

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM

VISUAL_CHATGPT_PREFIX = """Visual ChatGPT is designed to be able to assist with a wide range of text and visual related tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. Visual ChatGPT is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Visual ChatGPT is able to process and understand large amounts of text and images. As a language model, Visual ChatGPT can not directly read images, but it has a list of tools to finish different visual tasks. Each image will have a file name formed as "image/xxx.png", and Visual ChatGPT can invoke different tools to indirectly understand pictures. When talking about images, Visual ChatGPT is very strict to the file name and will never fabricate nonexistent files. When using tools to generate new image files, Visual ChatGPT is also known that the image may not be the same as the user's demand, and will use other visual question answering tools or description tools to observe the real image. Visual ChatGPT is able to use tools in a sequence, and is loyal to the tool observation outputs rather than faking the image content and image file name. It will remember to provide the file name from the last tool observation, if a new image is generated.

Human may provide new figures to Visual ChatGPT with a description. The description helps Visual ChatGPT to understand this image, but Visual ChatGPT should use tools to finish following tasks, rather than directly imagine from the description.

Overall, Visual ChatGPT is a powerful visual dialogue assistant tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. 


TOOLS:
------

Visual ChatGPT  has access to the following tools:"""

VISUAL_CHATGPT_FORMAT_INSTRUCTIONS = """To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
{ai_prefix}: [your response here]
```
"""

VISUAL_CHATGPT_SUFFIX = """You are very strict to the filename correctness and will never fake a file name if it does not exist.
You will remember to provide the image file name loyally if it's provided in the last tool observation.

Begin!

Previous conversation history:
{chat_history}

New input: {input}
Since Visual ChatGPT is a text language model, Visual ChatGPT must use tools to observe images rather than imagination.
The thoughts and observations are only visible for Visual ChatGPT, Visual ChatGPT should remember to repeat important information in the final response for Human. 
Thought: Do I need to use a tool? {agent_scratchpad}"""


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def cut_dialogue_history(history_memory, keep_last_n_words=500):
    if history_memory is None or len(history_memory) == 0:
        return history_memory
    tokens = history_memory.split()
    n_tokens = len(tokens)
    print(f"history_memory:{history_memory}, n_tokens: {n_tokens}")
    if n_tokens < keep_last_n_words:
        return history_memory
    paragraphs = history_memory.split('\n')
    last_n_tokens = n_tokens
    while last_n_tokens >= keep_last_n_words:
        last_n_tokens -= len(paragraphs[0].split(' '))
        paragraphs = paragraphs[1:]
    return '\n' + '\n'.join(paragraphs)

def text_marker(text, mark=None):
    result = []
    for letter in text:
        result.append((letter, mark))
    return result

model_map = {
    "Flan-t5": "flant5",
    "BLOOM": "bloom",
    "LLaMA": "llama"
}

model_map_rev = {
    "flant5": "Flan-t5",
    "bloom": "BLOOM",
    "llama": "LLaMA"
}

def highlight_maker(model_selected):
    result = []
    for model in model_map_rev.keys():
        print(model)
        if model is not model_map[model_selected]:
            result += text_marker(model, "_")
            result += text_marker("  ")
        else:
            result += text_marker(model, "<")
            result += text_marker("  ")
    return result
    
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


class ConversationBot:
    def __init__(self, model, run_type="normal"):
        # load_dict = {'VisualQuestionAnswering':'cuda:0', 'ImageCaptioning':'cuda:1',...}
        print(f"Initializing VisualChatAssistant")
        # self.llm = OpenAI(temperature=0)
        # self.llm = HuggingFaceHub(repo_id="google/flan-t5-small", model_kwargs={"tempearture":0.1, "max_new_tokens":200})

        self.run_type = run_type

        if run_type == "normal":
            self.load_model_normal(model)
        elif run_type == "test":
            self.load_model_test(model)

        self.make_chat_chain(model)

    def load_model_normal(self, model):
        if model == "flant5":
            model_id = "google/flan-t5-xxl"
            model_task = "text2text-generation"
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id, cache_dir='./')
        elif model == "bloom":
            # model_id = "bigscience/bloom-560m"
            model_id = "bigscience/bloom-7b1"
            model_task = "text-generation"
            model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir='./')
        elif model == "llama":
            model_id = "decapoda-research/llama-7b-hf"
            model_task = "text-generation"
            model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir='./')

        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir='./')

        pipe = pipeline(
            model_task,
            model=model,
            tokenizer=tokenizer,
            max_length=200
        )

        self.llm = HuggingFacePipeline(pipeline=pipe)

    def load_model_test(self, model):
        if model == "flant5":
            self.llm = HuggingFaceHub(repo_id="google/flan-t5-xxl",
                                        # model_kwargs={"temperature": 0.1, "max_new_tokens": 200})
                                        model_kwargs={"temperature": 0.1})
        elif model == "bloom":
            self.llm = HuggingFaceHub(repo_id="bigscience/bloom-7b1",
                                        model_kwargs={"temperature": 0.1})
        elif model == "llama":
            self.llm = HuggingFaceHub(repo_id="decapoda-research/llama-13b-hf",
                                        model_kwargs={"temperature": 0.1})

    def choose_memory(self, model):
        if model == "flant5":
            self.memory = ConversationBufferWindowMemory(k=4)
        elif model == "bloom":
            self.memory = ConversationBufferWindowMemory(k=4)
        elif model == "llama":
            self.memory = ConversationBufferWindowMemory(k=4)

    def make_chat_chain(self, model):
        template = """{history}
        Human:{human_input}
        Assistant:"""

        prompt = PromptTemplate(
            input_variables=["history", "human_input"],
            template=template
        )

        self.choose_memory(model)

        self.chat_chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            verbose=True,
            memory=self.memory
        )

    def set_model(self, model):
        print(f"Setting Model: {model}")

        if self.run_type == "normal":
            self.load_model_normal(model_map[model])
        elif self.run_type == "test":
            self.load_model_test(model_map[model])

        self.make_chat_chain(model)

        print(f"Setting Model: {model} successfully!")

        print(highlight_maker(model))

        return highlight_maker(model)

    def run_text(self, text, state):
        print("="*20+"run_text"+"="*20)
        print("inputs:", text, state)
        print(f">Previous memory:\n{self.memory}")
        response = self.chat_chain.predict(human_input=text)
        print(f">Current memory:\n{self.memory}")
        state += [(text, response)]
        print("Outputs:", state)
        return state, state


def change_model(modelname):
    bot.set_model(modelname)
    
    
def switch_note_maker():
    return text_marker("Switching Model")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', dest="model", type=str, default="flant5", choices=["flant5", "bloom", "llama"],
                        help="What model do you use? Choose one from above.")
    parser.add_argument('-t', "--run-type", dest="run_type", type=str, default="normal", choices=["normal", "test"],
                        help="Normal for local model, test for API(quick debug, For develop only).")
    args = parser.parse_args()
    bot = ConversationBot(model=args.model, run_type=args.run_type)
    with gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}") as demo:
        gr.Markdown(
        """
        # Chat bot!
        """)
        chatbot = gr.Chatbot(elem_id="chatbot", label="Chat Bot")
        state = gr.State([])
        with gr.Row():
            with gr.Column(scale=0.80):
                txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter, or upload an image").style(
                    container=False)
            with gr.Column(scale=0.20, min_width=0):
                clear = gr.Button("Clear")
        gr.Markdown(
        """
        *Note:* **Choose the model and click "Change Model" to change the model you use. It may takes several minutes to change model.**
        """)
        with gr.Row():
            with gr.Column():
                # switch_input = gr.Radio(choices=["Flan-t5", "BLOOM", "LLaMA"],
                switch_input = gr.Radio(choices=["Flan-t5", "LLaMA"],
                                                label="Switch Model", value=model_map_rev[args.model])
            # with gr.Column(scale=0.10, min_width=0):
            #     switch = gr.Button("Change model")
            with gr.Column():
                hightlight_text = gr.HighlightedText(
                        label="Model now",
                        combine_adjacent=True,
                    ).style(color_map={"<": "green", "_": "gray"})
        gr.Markdown(
        """
        # *Note:* **There are some trouble with BLOOM, Don't use BLOOM now, or it may let the server stuck.**
        """)

        switch_input.change(switch_note_maker, None, outputs=hightlight_text)
        switch_input.change(bot.set_model, inputs=switch_input, outputs=hightlight_text)
        # with gr.Column(scale=0.15, min_width=0):
            #     btn = gr.UploadButton("Upload", file_types=["image"])

        txt.submit(bot.run_text, [txt, state], [chatbot, state])
        txt.submit(lambda: "", None, txt)
        # btn.upload(bot.run_image, [btn, state, txt], [chatbot, state, txt])
        clear.click(bot.memory.clear)
        clear.click(lambda: [], None, chatbot)
        clear.click(lambda: [], None, state)
    # with gr.Column(scale=0.15, min_width=0):
    #     switch = gr.inputs.Dropdown(choices=["Flan-t5", "BLOOM", "LLaMA"],
    #                                 label="Switch Model", onchange=bot.set_model)
    # demo2 = gr.Interface(bot.set_model, [gr.inputs.Dropdown(choices=["Flan-t5", "BLOOM", "LLaMA"]), "text"], "text")
    # demoall = gr.TabbedInterface(demo, demo2)
    # demo.launch(server_port=8899, share=True, auth=("username", "llm-test-service"))
    demo.launch(server_name="0.0.0.0", server_port=8899)
