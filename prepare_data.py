import pandas as pd
from datasets import DatasetDict, Dataset
from sklearn.model_selection import train_test_split

# 1. Carregar o dataset do arquivo JSON Lines
# O Pandas é útil para ler o JSONL facilmente em um DataFrame
try:
    df = pd.read_json('data/jokes_bigger_dataset.jsonl', lines=True)
    print(f"Dataset carregado com {len(df)} exemplos.")
except FileNotFoundError:
    print("Erro: O arquivo 'data/jokes_dataset.jsonl' não foi encontrado. Certifique-se de que ele está na mesma pasta do script.")
    exit()

# 2. Dividir o DataFrame em conjuntos de treinamento e validação
# Usaremos sklearn.model_selection.train_test_split para isso
# 80% para treinamento, 20% para validação
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42) # random_state garante que a divisão seja a mesma cada vez que você rodar

print(f"Dataset de Treinamento: {len(train_df)} exemplos.")
print(f"Dataset de Validação: {len(val_df)} exemplos.")

# 3. Converter DataFrames do Pandas para objetos Dataset do Hugging Face
# Isso é importante porque a biblioteca `datasets` (e o `Trainer`) espera este formato
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Opcional: Combinar em um DatasetDict, que é a forma padrão do Hugging Face para gerenciar splits
# Isso facilita passar para o Trainer mais tarde
raw_datasets = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset
})

print("\nPrévia dos 3 primeiros exemplos do Dataset de Treinamento:")
print(raw_datasets['train'].to_pandas().head(3).to_markdown(index=False))

print("\nPrévia dos 3 primeiros exemplos do Dataset de Validação:")
print(raw_datasets['validation'].to_pandas().head(3).to_markdown(index=False))

# Opcional: Salvar os datasets divididos (útil para grandes datasets ou para carregar em outro script)
raw_datasets['train'].save_to_disk('data/jokes_train_dataset')
raw_datasets['validation'].save_to_disk('data/jokes_val_dataset')

print("\nDataset dividido e pronto para o próximo passo (Tokenização e Treinamento)!")