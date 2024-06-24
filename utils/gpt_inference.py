import json
from openai import AzureOpenAI
from utils.load_data_util import load_json_file
from datasets import load_dataset
import time
import re


AZURE_OPENAI_KEY = ""


class GPTInference:
    def __init__(self):
        self.client = AzureOpenAI(
            azure_endpoint="",
            api_key=AZURE_OPENAI_KEY,
            api_version="")

    def post_process(self, text):
        match = re.search(r"(?i)(?<=\banswer:\s).*", text)
        if match:
            return match.group(0)
        else:
            return text

    def predict(self, prompt, temperature=0, max_tokens=1000, retry=3, delay=5):
        message_text = [
            {"role": "user", "content": prompt},
        ]
        for attempt in range(retry):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",  # model = "deployment_name"
                    messages=message_text,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=12345,
                )
                break
            except:
                time.sleep(delay)
        response = self.post_process(response.choices[0].message.content)
        return response

    def predict_close_book(self, question, num_demo=16):
        # demo = load_json_file("")
        prompt = ("Here are some examples of questions and their corresponding answer, each with a 'Question' field and an 'Answer' field. "
                  "Answer the question directly and don't output other thing. ")
        for item in self.demo[:num_demo]:
            prompt += f"Question: {item['question']} Answer: {item['short_answers'][0]}\n"
        prompt += f"Question: {question} Answer: "
        answer = self.predict(prompt)
        return answer

    def predict_nq(self, context, question, titles):
        titles = ['"' + title + '"' for title in titles]
        prompt = (f"Go through the following context and then extract the answer of the question from the context. "
                  f"Answer the question directly. Your answer should be very concise. "
                  f"The context is a list of Wikipedia documents, ordered by title: {titles}. "
                  f"Each Wikipedia document contains a 'title' field and a 'text' field. "
                  f"The context is: {context}. "
                  f"The question: {question}. ")
        long_answer = self.predict(prompt)
        short_answer = self.extract_answer(question, long_answer)
        return long_answer, short_answer

    def predict_hotpotqa(self, context, question, titles):
        prompt = (f"Go through the following context and then answer the question "
                  f"The context is a list of Wikipedia documents titled: {titles}. "
                  f"There are two types of questions: comparison questions, which require a yes or no answer or a selection from two candidates, "
                  f"and general questions, which demand a concise response. "
                  f"The context is: {context}. "
                  f"Find the useful documents from the context, then answer the question: {question}."
                  f"For general questions, you should use the exact words from the context as the answer to avoid ambiguity. "
                  f"Answer the question directly and don't output other thing.  ")
        long_answer = self.predict(prompt)
        short_answer = self.extract_answer(question, long_answer)
        return long_answer, short_answer

    def generate_demo_examples(self, num_demo=8):
        """
        Generate in-context examples to extract the short answer from the long answer.
        """
        if num_demo == 0:
            return ""
        demo_data = load_dataset("TIGER-Lab/LongRAG", "answer_extract_example")['train']
        demo_prompt = "Here are some examples: "
        for item in demo_data.select(range(num_demo)):
            for answer in item["answers"]:
                demo_prompt += f"Question: {item["question"]}\nLong Answer: {item["long_answer"]}\nShort Answer: {answer}\n\n"
        return demo_prompt

    def extract_answer(self, question, long_answer):
        """
        Extract the final short answer from the long answer.
        """
        prompt = "As an AI assistant, you have been provided with a question and its long answer. " \
                 "Your task is to derive a very concise short answer, extracting a substring from the given long answer. " \
                 "Short answer is typically an entity without any other redundant words." \
                 "It's important to ensure that the output short answer remains as simple as possible.\n\n"
        prompt += self.generate_demo_examples(num_demo=8)
        prompt += f"Question: {question}\nLong Answer: {long_answer}\nShort Answer: "
        short_answer = self.predict(prompt)
        return short_answer
