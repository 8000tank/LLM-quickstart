import csv
import datetime
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 初始化ChatOpenAI
chat = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=1,
    max_tokens=4095,
    model_kwargs={
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }
)


def gen_data(raw_content):
    """使用LangChain LCEL语法处理单个数据样例"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        你是中国古典哲学大师，尤其擅长周易的哲学解读。

        接下来，你收到的都是关于周易卦象的解释，你需要整理润色，并生成用于大模型训练的内容和格式。

        示例输入：

        师卦，此卦是异卦相叠，下卦为坎，上卦为坤。“师”指军队。坎为水、为险；坤为地、为顺，喻寓兵于农。兵凶战危，用兵乃圣人不得已而为之，但它可以顺利无阻碍地解决矛盾，因为顺乎形势，师出有名，故能化凶为吉。占得此卦，对于军事上率师出征非常有利，必无灾祸。师卦是天马出群之卦，以寡伏众之象。
        师卦位于讼卦之后，《序卦》之中这样解释道：“讼必有众起，故受之以师。师者，众也。”争讼的人越来越多，以致形成了军队。

        期待结果：

        content:"师卦"
        summary:"在周易中，师卦是一个极具深意的卦象，它由两个异卦相叠组成：下卦坎（水）和上卦坤（地）。这一卦象代表“师”，即军队，寓意着兵力和农力的结合。在这里，坎卦象征着水和险难，而坤卦象征着地和顺从，暗示着通过将军事力量安置于民间，可以在必要时顺利调动。

        师卦的核心哲学是：虽然兵力代表着危险和战争，但其使用应当是圣人不得已而为之的最后手段。在正确的情况下，军事力量可以顺应形势，将危险转化为吉祥。因此，在军事策略上，此卦象征着出征将会顺利，无灾祸。

        师卦紧随讼卦（争讼卦），在《序卦》中解释为“讼必有众起，故受之以师”。这意味着争端激化至众多人群的参与，形成了类似军队的集体力量。"

        返回格式要求：
        
        content:[卦名]
        summary:[内容]
        """),
        ("human", "{input}")
    ])

    output_parser = StrOutputParser()
    chain = prompt | chat | output_parser
    return chain.invoke({"input": raw_content})


def dataset_parser(ai_message_content):
    """解析GPT生成的内容，提取content和summary"""
    content_start = ai_message_content.find('content:"') + len('content:"')
    content_end = ai_message_content.find('"\nsummary:')
    summary_start = ai_message_content.find('summary:"') + len('summary:"')
    summary_end = ai_message_content.rfind('"')

    content = ai_message_content[content_start:content_end].strip()
    summary = ai_message_content[summary_start:summary_end].strip()

    return content, summary


def generate_question_summary_pairs(content, summary):
    """生成30对提问和总结的配对"""
    question_templates = [
        "{}代表什么？",
        "周易中的{}含义是什么？",
        "请解释一下{}。",
        "{}在周易中是什么象征？",
        "周易{}的深层含义是什么？",
        "周易的{}讲述了什么？",
        "{}是怎样的一个卦象？",
        "{}在周易中怎样表达其象征的概念？",
        "{}的基本意义是什么？",
        "周易中{}的解释是什么？",
        "{}在周易中代表了哪些方面？",
        "{}涉及哪些哲学思想？",
        "周易中{}的象征意义是什么？",
        "{}的主要讲述内容是什么？",
        "周易{}的核心思想是什么？",
        "在周易中，{}象征着什么？",
        "请描述{}的含义。",
        "{}在周易哲学中扮演什么角色？",
        "{}的文化意义是什么？",
        "从周易的角度看，{}有何意义？",
        "在周易中，{}的象征作用是什么？",
        "{}对现代人有什么启发？",
        "{}象征着什么样的力量？",
        "{}的主要寓意是什么？",
        "{}在古代社会中的应用是什么？",
        "周易中的{}如何解读？",
        "{}的智慧是什么？",
        "从哲学角度，如何理解{}？",
        "解释一下{}的价值。",
        "{}与其他卦象的关系是什么？"
    ]

    questions = [template.format(content) for template in question_templates]
    return [(question, summary) for question in questions]


def main():
    """主函数：处理原始数据并生成训练数据集"""
    # # 确保 data 目录存在
    # if not os.path.exists('data'):
    #     os.makedirs('data')

    # 解析原始数据
    raw_content_data = []
    with open('chatglm/data/raw_data.txt', 'r', encoding='utf-8') as file:
        content = file.read()
        data_samples = content.split('\n\n')
        for sample in data_samples:
            cleaned_sample = sample.strip()
            if cleaned_sample:
                raw_content_data.append(cleaned_sample)

    # 创建输出文件
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chatglm/data/zhouyi_dataset_{timestamp}.csv"

    # 生成数据集
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['content', 'summary'])

        for raw_content in raw_content_data:
            ai_message_content = gen_data(raw_content)
            content, summary = dataset_parser(ai_message_content)

            print("Content:", content)
            print("Summary:", summary)

            pairs = generate_question_summary_pairs(content, summary)
            for pair in pairs:
                writer.writerow(pair)


if __name__ == "__main__":
    main()
