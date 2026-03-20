from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers
from tokenizers.trainers import BpeTrainer

def train_tokenizer():
    print("Loading dataset samples from 6 sources for tokenizer training...")
    
    # 1. TinyStories
    ds_tiny = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    # 2. Cosmopedia Stories (General Version)
    ds_stories = load_dataset("HuggingFaceTB/cosmopedia", name="stories", split="train", streaming=True)
    # 3. Cosmopedia-v2
    ds_cosmo = load_dataset("HuggingFaceTB/smollm-corpus", name="cosmopedia-v2", split="train", streaming=True)
    # 4. FineWeb-Edu
    ds_fineweb = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    # 5. ProofWriter
    ds_proof = load_dataset("tasksource/proofwriter", split="train", streaming=True)
    # 6. TinyCodes
    ds_codes = load_dataset("nampdn-ai/tiny-codes", split="train", streaming=True).filter(lambda x: x["programming_language"] == "Python")

    def batch_iterator(batch_size=1000):
        buffer = []
        
        datasets = [
            (ds_tiny, "TinyStories", lambda x: x["text"]),
            (ds_stories, "Cosmo-Stories", lambda x: x["text"]),
            (ds_cosmo, "Cosmo-v2", lambda x: x["text"]),
            (ds_fineweb, "FineWeb-Edu", lambda x: x["text"]),
            (ds_proof, "ProofWriter", lambda x: f"Context:\n{x['theory']}\n\nQuestion: {x['question']}\nAnswer: {str(x['answer'])}"),
            (ds_codes, "TinyCodes", lambda x: f"Question:\n{x['prompt']}\n\nCode:\n{x['response']}")
        ]

        for ds, name, mapper in datasets:
            print(f"Sampling {name} (5k)...")
            count = 0
            for item in ds:
                if count >= 5000: break
                buffer.append(mapper(item))
                count += 1
            
        print(f"Collected total {len(buffer)} samples. Training tokenizer...")
        
        for i in range(0, len(buffer), batch_size):
            yield buffer[i : i + batch_size]

    # Initialize Tokenizer (BPE)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = BpeTrainer(
        vocab_size=16384,
        special_tokens=["<UNK>", "<PAD>", "<BOS>", "<EOS>"],
        show_progress=True
    )

    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    
    # Save
    tokenizer.save("tokenizer.json")
    print("Tokenizer saved to tokenizer.json")
    
    # Verification
    print("\nVerification:")
    output = tokenizer.encode("Hello, how are you today?")
    print(f"Encoded IDs: {output.ids}")
    print(f"Decoded: {tokenizer.decode(output.ids)}")
    print(f"Vocab Size: {tokenizer.get_vocab_size()}")

if __name__ == "__main__":
    train_tokenizer()
