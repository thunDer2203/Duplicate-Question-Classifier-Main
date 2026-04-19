import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import os
import re
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix



def load_model():
    results_dir = './results'

    if not os.path.exists(results_dir):
        raise FileNotFoundError("❌ 'results' folder not found. Run train.py first.")

    checkpoints = [d for d in os.listdir(results_dir) if d.startswith('checkpoint-')]

    if not checkpoints:
        raise FileNotFoundError("❌ No checkpoints found.")

    checkpoint_numbers = [
        int(re.search(r'checkpoint-(\d+)', cp).group(1))
        for cp in checkpoints
    ]

    latest_checkpoint = max(checkpoint_numbers)
    model_path = f'{results_dir}/checkpoint-{latest_checkpoint}'

    print(f"✅ Loading checkpoint: {model_path}")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(model_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    return tokenizer, model, device



def predict_duplicate(q1, q2, tokenizer, model, device):
    inputs = tokenizer(
        q1,
        q2,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors='pt'
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)

    return probs[0][1].item()

def run_tests(test_cases_file='./test_cases.json'):

    try:
        with open(test_cases_file, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
        print(f"📂 Loaded {len(test_cases)} test cases\n")
    except:
        print("❌ Test cases file not found")
        return

    tokenizer, model, device = load_model()
    os.makedirs('testing', exist_ok=True)

    y_true = []
    y_pred = []

    for i, test in enumerate(test_cases, 1):
        prob = predict_duplicate(test["q1"], test["q2"], tokenizer, model, device)
        prediction = "Duplicate" if prob > 0.5 else "Not Duplicate"

        true_label = 1 if test['expected'] == "Duplicate" else 0
        pred_label = 1 if prediction == "Duplicate" else 0

        y_true.append(true_label)
        y_pred.append(pred_label)

        print(f"{i}. {prediction} ({prob:.2f})")

    # Metrics
    print("\n📊 TEST CASE METRICS:")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred):.4f}")


def test_on_real_data(csv_file="quora.csv", sample_size=1000):
    import pandas as pd
    import os
    from datetime import datetime

    print("📂 Loading dataset...")
    df = pd.read_csv(csv_file)

    df = df[['question1', 'question2', 'is_duplicate']].dropna()
    df = df.sample(sample_size, random_state=42)

    print(f"🔍 Testing on {len(df)} samples...\n")

    tokenizer, model, device = load_model()

    y_true = []
    y_pred = []

    os.makedirs("testing", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = f"testing/detailed_results_{timestamp}.txt"

    with open(file_path, "w", encoding="utf-8") as f:

        f.write("="*80 + "\n")
        f.write("Duplicate Question Classifier - Results\n")
        f.write(f"Time: {datetime.now()}\n")
        f.write("="*80 + "\n\n")

        for i, row in enumerate(df.itertuples(), 1):

            prob = predict_duplicate(
                row.question1,
                row.question2,
                tokenizer,
                model,
                device
            )

            prediction = "Duplicate" if prob > 0.5 else "Not Duplicate"
            expected = "Duplicate" if row.is_duplicate == 1 else "Not Duplicate"

            pred_label = 1 if prob > 0.5 else 0

            y_true.append(row.is_duplicate)
            y_pred.append(pred_label)

            status = "PASS" if prediction == expected else "FAIL"

            # Write to file
            f.write(f"Test Case {i}\n")
            f.write("-"*60 + "\n")
            f.write(f"Q1: {row.question1}\n")
            f.write(f"Q2: {row.question2}\n")
            f.write(f"Probability: {prob:.4f}\n")
            f.write(f"Prediction: {prediction}\n")
            f.write(f"Expected: {expected}\n")
            f.write(f"Status: {status}\n\n")

            if i % 100 == 0:
                print(f"Processed {i} samples...")  

        # Metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        f.write("="*80 + "\n")
        f.write("METRICS\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall: {rec:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n")

    print(f"\n📁 Detailed results saved to: {file_path}")

if __name__ == "__main__":
    # 👉 Choose ONE:

    # run_tests()              # for your custom test_cases.json
    test_on_real_data()        # for real dataset