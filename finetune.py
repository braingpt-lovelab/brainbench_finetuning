import os
import random
import argparse
import math
import pickle
import time
import json
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import SchedulerType, AdamW, get_scheduler
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from accelerate import Accelerator

from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from peft import PeftConfig, PeftModel
from torch.utils.data import DataLoader


# accelerator = Accelerator()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = accelerator.device
random.seed(1)
torch.manual_seed(1)


def ddp_setup(rank: int, world_size: int, master_port: str):
    """
    Args:
        rank: Unique identifier of each process
       world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = master_port
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def logging(s, logfile, logging_=True, log_=True):
    if logging_:
        print(s)
    if log_:
        with open(logfile, 'a+') as f_log:
            f_log.write(s + '\n')


def tokenize(element, tokenizer, args):
    outputs = tokenizer(
        element["text"],
        truncation=True,
        max_length=args.chunk_size,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    input_lengths = []
    return outputs


def collate_fn(batch):
    input_ids = [sample["input_ids"] for sample in batch]
    attention_masks = [sample["attention_mask"] for sample in batch]
    labels = pad_sequence(input_ids, batch_first=True, padding_value=-1)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_masks.to(device),
        "labels": labels.to(device),
    }


def save_checkpoint(LLM, tokenizer, outputdir, epoch):
    fulloutput = os.path.join(outputdir, "checkpoint.{}".format(epoch))
    os.system(f"mkdir -p {fulloutput}")
    # save tokenizer
    tokenizer.save_pretrained(fulloutput)
    # save model
    LLM.module.save_pretrained(fulloutput)


def main(rank, args, world_size):
    ## Setup DDP
    ddp_setup(rank, world_size, args.master_port)
    print(f"rank: {rank}")

    # Save model configuration
    with open(os.path.join(args.outputdir, 'model_config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Load huggingface dataset
    dataset = load_dataset(args.data_path)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, cache_dir="/datadrive1/ken/.cache/huggingface/hub")
    tokenized_dataset = dataset.map(
        tokenize,
        fn_kwargs={"tokenizer": tokenizer, "args": args},
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    logging("Loading {} samples for training".format(len(tokenized_dataset["train"])), args.logfile)
    tokenized_dataset.set_format("torch")
    train_dataloader = DataLoader(
        tokenized_dataset["train"],
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        sampler=DistributedSampler(tokenized_dataset["train"]),
        shuffle=False,
    )
    valid_dataloader = DataLoader(
        tokenized_dataset["validation"],
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        sampler=DistributedSampler(tokenized_dataset["validation"]),
    )

    # Define model
    with open(args.lora_config) as fin:
        lora_config = json.load(fin)
    os.system("cp {} {}".format(args.lora_config, os.path.join(args.outputdir, 'lora_config.json')))
    LLM = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, cache_dir="/datadrive1/ken/.cache/huggingface/hub")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_config["lora_rank"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        target_modules=lora_config["lora_module"],
    )
    LLM = get_peft_model(LLM, peft_config)
    LLM.print_trainable_parameters()
    LLM = LLM.to(device)
    LLM = DDP(LLM, device_ids=[rank])

    ## Initialise criterion and optimiser
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    ## Optimiser
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in LLM.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in LLM.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    num_warmup_steps = args.num_warmup_steps * max_train_steps

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )
    # LLM, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
    #     LLM, optimizer, train_dataloader, valid_dataloader)

    logging("Start training", args.logfile)
    # Training loop
    best_val_loss = 10000
    trainsize = len(train_dataloader)
    for epoch in range(args.num_train_epochs):
        start = time.time()
        optimizer.zero_grad()
        for i, batch in enumerate(train_dataloader):
            logits = LLM(**batch).logits[:, :-1]
            labels = batch["labels"][:, 1:]
            loss = criterion(logits.view(-1, logits.size(-1)), labels.reshape(-1))
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            # accelerator.backward(loss)

            if (i + 1) % args.gradient_accumulation_steps == 0:
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            if (i + 1) % args.log_interval == 0:
                elasped_time = time.time() - start
                PPL = math.exp(loss.item() * args.gradient_accumulation_steps)
                logging(f"Epoch {epoch} | Batch {i}/{trainsize} | PPL: {PPL} | time {elasped_time}", args.logfile)
            
            if args.save_interval > 0 and (i + 1) % args.save_interval == 0:
                # Evaluate every args.save_interval steps
                LLM.eval()
                with torch.no_grad():
                    val_loss = evaluate(args, LLM, valid_dataloader, criterion)
                    val_ppl = math.exp(val_loss)
                    current_lr = optimizer.param_groups[0]["lr"]
                    torch.distributed.reduce(val_loss, 0)
                    # Save models
                    if rank == 0:
                        logging(f"Epoch {epoch} | Validation PPL: {val_ppl/world_size} | Learning rate: {current_lr}", args.logfile)
                        if val_loss < best_val_loss:
                            ckpt_path = os.path.join(args.outputdir, "checkpoint.{}_{}".format(epoch, (i + 1)))
                            logging(f"Save checkpoint to {ckpt_path}", args.logfile)
                            save_checkpoint(LLM, tokenizer, args.outputdir, f"{epoch}_{(i+1)}")
                LLM.train()
        # Evaluate again at the end of epoch
        LLM.eval()
        with torch.no_grad():
            val_loss = evaluate(args, LLM, valid_dataloader, criterion)
            val_ppl = math.exp(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            torch.distributed.reduce(val_loss, 0)
            # Save models
            if rank == 0:
                logging(f"End of epoch {epoch} | Validation PPL: {val_ppl/world_size} | Learning rate: {current_lr}", args.logfile)
                if val_loss < best_val_loss:
                    ckpt_path = os.path.join(args.outputdir, "checkpoint.{}".format(epoch))
                    logging(f"Save checkpoint to {ckpt_path}", args.logfile)
                    save_checkpoint(LLM, tokenizer, args.outputdir, f"{epoch}")
        LLM.train()


def evaluate(args, LLM, valid_dataloader, criterion):
    total_tokens = 0
    total_loss = 0.
    for i, batch in enumerate(valid_dataloader):
        with torch.cuda.amp.autocast():
            logits = LLM(**batch).logits[:, :-1]
            labels = batch["labels"][:, 1:]
            loss = criterion(logits.view(-1, logits.size(-1)), labels.reshape(-1))
            ntokens = (batch["attention_mask"][:, 1:] == 1).sum()
            total_tokens += ntokens
            total_loss += loss * ntokens
    return total_loss / total_tokens


if __name__ == "__main__":
    ## Parameter groups
    parser = argparse.ArgumentParser(description="LLM finetuning")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./hf_models",
        help="Path to the model file",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./hf_models",
        help="Path to the train data file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=4096,
        help="maximum number of tokens in each sample"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0, help="Weight decay."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=float, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--logfile",
        type=str,
        default='./log.txt',
        help="Path to the log file",
    )
    parser.add_argument(
        "--outputdir",
        type=str,
        default='./exp/clip_vlm',
        help="Path to the output dir",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="log interval",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=0,
        help="Saving interval",
    )
    parser.add_argument(
        "--master_port",
        type=str,
        default='12355',
        help="Master port number",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare responses or not",
    )
    parser.add_argument(
        "--classhead",
        type=str,
        default="none",
        help="Use classification head as the output",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora_config",
        type=str,
        default="config/lora_config.json",
        help="LoRA configuration",
    )
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    print(world_size)
    mp.spawn(main, args=(args, world_size,), nprocs=world_size)
    # main(0, args, world_size)
