import jieba
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import os
from django.shortcuts import render
from django.http import JsonResponse
import read_kb

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 强制使用 CPU

# 加载预训练的问答模型
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

# 全局变量，用于存储问答对和反馈数据
qa_pairs = {}
feedback_data = []

# 定义文件夹路径
folder_path = r"D:\FOR_NLP\datalist"

# 读取知识库
knowledge_base = read_kb.read_knowledge_base(folder_path)

# 自然语言处理技术 - 词法分析
def lexical_analysis(question):
    return jieba.lcut(question)

# 问答模型 - 结合规则匹配和预训练模型
def answer_generation(question, context):
    max_length = 512
    stride = 256  # 滑动窗口的步长
    inputs = tokenizer(question, context, return_tensors='pt', max_length=max_length, truncation=False)
    input_ids = inputs["input_ids"].tolist()[0]
    answers = []
    for i in range(0, len(input_ids), stride):
        chunk_input_ids = input_ids[i:i + max_length]
        chunk_inputs = {
            "input_ids": torch.tensor([chunk_input_ids]),
            "attention_mask": torch.tensor([[1] * len(chunk_input_ids)])
        }
        outputs = model(**chunk_inputs)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits
        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(chunk_inputs['input_ids'][0][answer_start:answer_end]))
        if answer.strip() != "":
            answers.append(answer)
    if not answers:
        return "很抱歉，我没有找到相关答案。你可以提供更多反馈帮助我学习。"
    return " ".join(answers)

# 人类反馈机制实现
def collect_feedback(question, answer, user_feedback):
    feedback_data.append((question, answer, user_feedback))
    if user_feedback.lower() == "不满意":
        new_answer = input("请提供正确的答案，帮助我学习：")
        qa_pairs[question] = new_answer
        return "感谢你的反馈，我已经学习到新的知识啦。下次遇到这个问题我会用你提供的答案回答。"
    return "感谢你的反馈！"

def index(request):
    return render(request, 'index.html')

def get_answer(request):
    if request.method == 'POST':
        question = request.POST.get('question')
        words = lexical_analysis(question)
        answer = answer_generation(question, knowledge_base)
        return JsonResponse({
            'words': words,
            'answer': answer
        })
    return JsonResponse({'error': 'Invalid request method'})

def submit_feedback(request):
    if request.method == 'POST':
        question = request.POST.get('question')
        answer = request.POST.get('answer')
        user_feedback = request.POST.get('feedback')
        result = collect_feedback(question, answer, user_feedback)
        return JsonResponse({'message': result})
    return JsonResponse({'error': 'Invalid request method'})