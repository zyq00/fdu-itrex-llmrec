import env_config
from transformers import TrainingArguments
from intel_extension_for_transformers.neural_chat.config import (
    ModelArguments,
    DataArguments,
    FinetuningArguments,
    TextGenerationFinetuningConfig,
)
from intel_extension_for_transformers.neural_chat.chatbot import finetune_model

def ft_reranker(datasize):
    #model_name="Qwen/Qwen-7B-Chat"
    #model_name="Intel/neural-chat-7b-v3-3"
    model_name_or_path=env_config.llm_dir+env_config.qwen_chat
    model_args = ModelArguments(model_name_or_path=model_name_or_path,trust_remote_code=True)
    data_args = DataArguments(train_file=env_config.movie_data_dir+f"alpaca_data_train_{datasize}.json")
    training_args = TrainingArguments(
        output_dir=env_config.llm_dir+env_config.model_rerank+f"-{datasize}",
        do_train=True,
        do_eval=False,
        num_train_epochs=3,
        overwrite_output_dir=True,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=1,
        save_strategy="no",
        log_level="info",
        save_total_limit=2,
        bf16=False,
        logging_dir='./logs',
        logging_steps=10,
    )
    finetune_args = FinetuningArguments()
    finetune_args.lora_target_modules = ['w1', 'w2', 'c_proj', 'c_attn']
    finetune_cfg = TextGenerationFinetuningConfig(
                model_args=model_args,
                data_args=data_args,
                training_args=training_args,
                finetune_args=finetune_args,
            )
    finetune_model(finetune_cfg)

#用专业影评微调对话模型 
def ft_chatbot():
    print('')
if __name__ == '__main__':
    ft_reranker(3000)