import transformers

from transformers import Trainer, AutoConfig, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

from dataloaders import TextDataset
from dataloaders import (
    sft_data_collactor,
    offline_ppo_data_collactor,
    weighted_sft_data_collactor,
)
from arguments import CustomTrainingArguments

from trainers import SFTWeightedWithKLTrainer, OfflineWeightedPolicyTrainer

from utils import print_rank_0, read_json_or_jsonl_data, set_special_tokens
from utils import DEFAULT_PAD_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_UNK_TOKEN

import os
import torch

from transformers import BitsAndBytesConfig

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, pipeline

model_dir = "ckpts/sentiment/distilbert-base-uncased-finetuned-sst-2-english"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    # debug: 检查权重 shape
    emb_shape = model.base_model.embeddings.word_embeddings.weight.shape
    print("✅ Loaded sentiment model. Embedding shape new:", emb_shape)

    sentiment_classifier = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=-1
    )
except Exception as e:
    print("❌ Failed to load sentiment model:", e)
    raise

semantic_model = SentenceTransformer("ckpts/sbert/all-MiniLM-L6-v2")


def get_train_dataset(args):
    all_train_data = []
    for train_data_path in args.train_data_path:
        train_data = read_json_or_jsonl_data(train_data_path)
        all_train_data.extend(train_data)

    if args.debug_mode:
        print_rank_0(f">>> check loaded data:")
        print_rank_0(f">>> {all_train_data[0]}")

    train_set = TextDataset(all_train_data)
    return train_set


def train():
    parser = transformers.HfArgumentParser(CustomTrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]
    print_rank_0(args)

    # load data
    # ---------------------------------------------------------------------------------
    train_dataset = get_train_dataset(args)

    # setup model
    # ---------------------------------------------------------------------------------
    print_rank_0(f"Begin loading model from {args.model_name_or_path}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_cache=False
    )
    if hasattr(model, "ref_model"):
        del model.ref_model


    if args.train_method in ["SFTwithKL", "OfflinePO"] and args.ref_model_name_or_path:
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.ref_model_name_or_path,
            trust_remote_code=True,
            use_cache=False
        )
        if hasattr(ref_model, "ref_model"):
            del ref_model.ref_model
        for param in ref_model.parameters():
            param.requires_grad = False
        model.ref_model = ref_model


    print_rank_0(model)
    print_rank_0(f"Finished loading model from {args.model_name_or_path}")

    model.is_parallelizable = True
    model.model_parallel = True

    # setup tokenizer
    # ---------------------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.max_length,
        padding_side=args.padding_side,
        truncation_side=args.truncation_side,
        trust_remote_code=True,
    )

    model, tokenizer = set_special_tokens(model, tokenizer)
    # build trainer
    # ---------------------------------------------------------------------------------

    if args.train_method == "SFT":
        trainer = Trainer(
            model=model,
            processing_class=tokenizer,
            args=args,
            train_dataset=train_dataset,
            data_collator=lambda x: sft_data_collactor(args, x, tokenizer),
        )

    elif args.train_method == "SFTwithKL":
        trainer = SFTWeightedWithKLTrainer(
            model=model,
            processing_class=tokenizer,
            args=args,
            train_dataset=train_dataset,
            data_collator=lambda x: offline_ppo_data_collactor(args, x, tokenizer),
        )

    elif args.train_method == "OfflinePO":
        trainer = OfflineWeightedPolicyTrainer(sentiment_classifier=sentiment_classifier,
            semantic_model=semantic_model,
            model=model,
            processing_class=tokenizer,
            args=args,
            train_dataset=train_dataset,
            data_collator=lambda x: offline_ppo_data_collactor(args, x, tokenizer)
        )

    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    trainer.save_state()
    if hasattr(trainer.model, "ref_model"):
        del trainer.model.ref_model

    trainer.save_model(output_dir=args.output_dir)


if __name__ == "__main__":
    train()
