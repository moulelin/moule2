from datasets import load_dataset

def load_and_print_gsm8k():
    # 加载 gsm8k 数据集
    # "main" 是默认的配置名称
    print("正在加载数据集...")
    dataset = load_dataset("gsm8k", "main")
    
    # 获取训练集的前 5 个数据
    # dataset 结构通常为 {'train': ..., 'test': ...}
    top_5 = dataset['train'].select(range(5))
    
    print("\n=== GSM8K 前 5 个样本 ===\n")
    
    for i, example in enumerate(top_5):
        print(f"--- 样本 {i+1} ---")
        print(f"【问题 (Question)】:\n{example['question']}")
        print(f"\n【答案 (Answer)】:\n{example['answer']}")
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    load_and_print_gsm8k()