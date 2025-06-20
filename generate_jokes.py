import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# --- 1. Configurações ---
# Nome do modelo base que você usou para o fine-tuning
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3" # Certifique-se de que é o mesmo que você usou
# Caminho para o diretório onde o adaptador LoRA foi salvo
LORA_ADAPTER_PATH = "./data/results_jokes_model"

# --- 2. Configuração da Quantização (4-bit) ---
# Use a mesma configuração de quantização que você usou no treinamento
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

# --- 3. Carregar o Modelo Base Pré-treinado (quantizado) ---
print(f"Carregando modelo base: {MODEL_NAME}...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto", # Garante que o modelo use a GPU
    trust_remote_code=True,
)
# Opcional: Para inference, use_cache pode ser True novamente para desempenho
base_model.config.use_cache = True

# --- 4. Carregar o Adaptador LoRA e Mesclá-lo ao Modelo Base ---
print(f"Carregando adaptador LoRA de: {LORA_ADAPTER_PATH} e mesclando...")
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)

# Se quiser mesclar as camadas LoRA de forma permanente (requer mais VRAM, mas acelera a inferência e pode ser salvo)
# model = model.merge_and_unload() # Descomente se quiser mesclar. Para este teste, não é estritamente necessário.

# --- 5. Carregar o Tokenizador ---
print(f"Carregando tokenizador: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Modelo finetunado e tokenizador carregados com sucesso. Pronto para gerar piadas!")

# --- 6. Função de Geração de Piadas ---
def generate_joke(topic, instruction_prefix="Gere uma piada sobre o tópico"):
    # Formata o prompt exatamente como o modelo foi treinado
    prompt = f"<s>[INST] {instruction_prefix}: {topic} [/INST]"
    
    # Codifica o prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda") # Envia para a GPU

    # Gera a resposta
    with torch.no_grad(): # Desativa o cálculo de gradientes para inferência, economiza memória
        outputs = model.generate(
            **inputs,
            max_new_tokens=100, # Comprimento máximo da piada a ser gerada
            do_sample=True, # Amostragem para mais criatividade (menos determinístico)
            top_k=50, # Considera apenas as 50 palavras mais prováveis
            top_p=0.95, # Amostragem de núcleo: considera um conjunto de palavras cuja probabilidade acumulada é 0.95
            temperature=0.7, # Controla a aleatoriedade (maior = mais criativo/aleatório, menor = mais focado)
            num_return_sequences=1, # Gera apenas uma piada por vez
            eos_token_id=tokenizer.eos_token_id, # Define o token de fim de sequência
            pad_token_id=tokenizer.pad_token_id, # Define o token de padding
        )
    
    # Decodifica a saída
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extrai apenas a parte da piada (o que vem depois do [/INST])
    # O modelo foi treinado para gerar ' [/INST] {piada}</s>'
    start_index = decoded_output.find("[/INST]")
    if start_index != -1:
        joke = decoded_output[start_index + len("[/INST]"):].strip()
        # Remove o token de fim de sequência se ele não foi removido pelo skip_special_tokens
        if joke.endswith("</s>"):
            joke = joke[:-4].strip()
        return joke
    return decoded_output # Retorna tudo se não encontrar a marcação

# --- 7. Loop de Interação com o Usuário ---
if __name__ == "__main__":
    print("\n--- Gerador de Piadas Personalizadas ---")
    print("Digite um tópico para a piada (ou 'sair' para encerrar):")

    while True:
        user_topic = input("\nTópico: ")
        if user_topic.lower() == 'sair':
            print("Encerrando o gerador de piadas. Até a próxima!")
            break
        
        # Você pode experimentar com diferentes prefixes de instrução que usou no dataset
        # instruction_prefix = "Crie uma piada sobre" 
        # instruction_prefix = "Me faça rir com uma piada sobre"
        
        print(f"Gerando piada sobre '{user_topic}'...")
        joke = generate_joke(user_topic, instruction_prefix="Gere uma piada sobre o tópico")
        print(f"Piada: {joke}")