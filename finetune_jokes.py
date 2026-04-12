import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# --- 1. Configurações Principais ---
# Nome do modelo base que vamos usar
# Certifique-se de que este modelo existe no Hugging Face Hub
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3" # "mistralai/Mistral-7B-Instruct-v0.2" # Ou v0.3 se preferir

# Caminho para o seu dataset JSON Lines
DATASET_PATH = "data/jokes_bigger_dataset.jsonl"

# Nome da pasta onde o modelo finetunado será salvo
OUTPUT_DIR = "./data/results_jokes_bigger_model"

# --- 2. Configuração do QLoRA ---
# Define a configuração do LoRA para o fine-tuning
# r: rank das matrizes de adaptação (menor = menos memória, menor expressividade)
# lora_alpha: fator de escala (geralmente o dobro de r)
# target_modules: quais módulos do modelo terão adaptadores LoRA (comuns para atenção)
# lora_dropout: dropout para as camadas LoRA (para evitar overfitting)
# bias: tipo de bias (none é comum para LoRA)
# task_type: tipo da tarefa (CAUSAL_LM para modelos generativos)
peft_config = LoraConfig(
    r=16, # Rank: Um bom ponto de partida, ajuste se precisar de mais capacidade ou menos VRAM
    lora_alpha=32, # Alpha: Geralmente o dobro do rank
    target_modules=["q_proj", "v_proj"], # Módulos de atenção são os mais comuns para adaptar
    lora_dropout=0.05, # Adiciona um pouco de dropout para regularização
    bias="none",
    task_type="CAUSAL_LM", # Para modelos generativos (Decoder-only)
)

# --- 3. Configuração da Quantização (4-bit) ---
# Necessário para QLoRA para reduzir o consumo de VRAM
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, # Carrega o modelo em 4 bits
    bnb_4bit_quant_type="nf4", # Tipo de quantização (nf4 é recomendado para Transformers)
    bnb_4bit_compute_dtype=torch.float16, # dtype para cálculos (float16 é eficiente para GPUs modernas)
    bnb_4bit_use_double_quant=False, # Não usar double quantização para este projeto inicial
)

# --- 4. Carregar o Modelo e Tokenizador ---
print(f"Carregando modelo e tokenizador: {MODEL_NAME}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config, # Aplica a configuração de quantização
    device_map="auto", # Mapeia automaticamente o modelo para as GPUs disponíveis
    trust_remote_code=True, # Permite carregar código remoto (necessário para alguns modelos)
)
# Prepara o modelo para treinamento em k-bit (necessário para QLoRA)
model.config.use_cache = False # Desativa cache durante o treinamento para evitar problemas de memória
model.config.pretraining_tp = 1 # Otimização para modelos grandes, mantenha como 1

model = prepare_model_for_kbit_training(model) # Prepara o modelo para o treinamento k-bit

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
# Configura o tokenizador para ter um token de padding
# Essencial para que os lotes de diferentes tamanhos sejam processados corretamente
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Padding à direita é comum para modelos generativos

# Aplica as configurações LoRA ao modelo quantizado
model = get_peft_model(model, peft_config)
model.print_trainable_parameters() # Mostra quantos parâmetros serão treinados (deve ser uma pequena % do total)

# --- 5. Carregar e Preparar o Dataset ---
print(f"Carregando dataset de: {DATASET_PATH}...")
# A biblioteca datasets pode carregar diretamente de JSON Lines
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

# Dividir o dataset (datasets.Dataset possui um método de divisão)
dataset = dataset.train_test_split(test_size=0.2, seed=42) # 80% treino, 20% teste/validação
train_dataset = dataset["train"]
eval_dataset = dataset["test"] # Usamos 'test' para o split de avaliação

# Função para formatar o prompt para o modelo (instrução + input + output)
# Este é o formato que o modelo vai aprender a gerar
def formatting_prompts_func(examples):
    output_texts = []
    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        input_text = examples["input"][i]
        output_text = examples["output"][i]
        
        # Formato de prompt comum para modelos instruct (como Mistral)
        # O modelo aprenderá a "responder" ao prompt com a piada
        text = f"<s>[INST] {instruction}: {input_text} [/INST] {output_text}</s>"
        output_texts.append(text)
    return output_texts

# --- 6. Configuração dos Argumentos de Treinamento ---
# Define os hiperparâmetros e configurações do treinamento
training_arguments = TrainingArguments(
    output_dir=OUTPUT_DIR, # Diretório para salvar os checkpoints e logs
    num_train_epochs=3, # Número de épocas de treinamento (ajuste conforme o resultado)
    per_device_train_batch_size=4, # Tamanho do batch por GPU (4 é um bom começo para 3060 com QLoRA)
    gradient_accumulation_steps=1, # Acumular gradientes para simular um batch maior (se o per_device_train_batch_size for muito pequeno)
    optim="paged_adamw_8bit", # Otimizador otimizado para memória com 8 bits
    save_strategy="epoch", # Salvar checkpoints a cada época
    logging_steps=10, # Frequência de log (para ver a perda)
    learning_rate=2e-4, # Taxa de aprendizado (um bom ponto de partida para fine-tuning)
    fp16=True, # Usar precisão float16 (acelera e economiza memória)
    tf32=True, # Usar precisão tf32 (acelera ainda mais em GPUs Ampere+)
    max_grad_norm=0.3, # Cortar gradientes para evitar explosão
    warmup_ratio=0.03, # Proporção de passos para aquecimento da taxa de aprendizado
    lr_scheduler_type="constant", # Tipo de agendador da taxa de aprendizado
    report_to="tensorboard", # Relatar métricas para TensorBoard (para visualização do progresso)
    load_best_model_at_end=True, # Carregar o melhor modelo no final do treinamento
    metric_for_best_model="eval_loss", # Métrica para determinar o melhor modelo
    eval_strategy="epoch"
)

# --- 7. Configurar e Iniciar o SFTTrainer ---
print("Configurando SFTTrainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset, # Dataset de validação
    peft_config=peft_config,
    formatting_func=formatting_prompts_func, # Função para formatar os prompts
    args=training_arguments,
)

# Treinar o modelo
print("\nIniciando treinamento...")
trainer.train()

print("\nTreinamento concluído! Salvando o adaptador PEFT final...")
# Salvar apenas o adaptador LoRA (que é pequeno)
trainer.save_model(OUTPUT_DIR)

print(f"\nModelo finetunado salvo em: {OUTPUT_DIR}")
print("Você pode carregar o adaptador PEFT para inferência.")