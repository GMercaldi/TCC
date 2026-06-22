#Fine tuning do 2019_fr with 2019_cat

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score
from sklearn.model_selection import train_test_split

#Determinação dos pesos para o treinamento
pesos = torch.tensor([
        1.0, #Wheat
        1.0, #Maize
        1.0, #Sorghum
        1.0, #Barley
        1.0, #Rye
        1.0, #Oats
        1.0, #Grapes
        1.0, #Rapeseed
        1.0, #Sunflower
        1.0, #Potato
        1.0, #Pea
])

#Função para carregar o dataset de finetuning

def main():
    #carregando o datasetde finetuning por meio de pandas
    df = pd.read_csv('coco_full_2019_cat.csv.gz')
   
    # Definindo as classes
    selected_classes = {
            110: 'Wheat',
            120: 'Maize',
            140: 'Sorghum',
            150: 'Barley',
            160: 'Rye',
            170: 'Oats',
            330: 'Grapes',
            435: 'Rapeseed',
            438: 'Sunflower',
            510: 'Potato',
            770: 'Pea'
    }
    df = df[df["label"].isin(selected_classes.keys())]
    class_names = [selected_classes[k] for k in sorted(selected_classes.keys())]
    
    #Informando as features
    feature_cols = [c for c in df.columns if 'mean' in c or 'std' in c]
    X = df[feature_cols].values

    #Carregar o pickles encoder linear desenvolvido do MODELO TREINADO
    linear_encoder = joblib.load('linear_encoder_2019_fr.pkl')
    y = df['label'].map(linear_encoder).values
    n_classes = len(linear_encoder)
     
    #Train, test split
    X_train_np, X_test_np, y_train, y_test = train_test_split(
            X,
            y,
            test_size = 0.2,
            stratify = y,
            random_state = 42
    )


    #Carregar o pickles do scaler do MODELO TREINADO
    scaler = joblib.load('scaler_2019_fr.pkl')
    X_train_np = scaler.transform(X_train_np)
    X_test_np = scaler.transform(X_test_np)

    #Forçar GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device:{device}')

    #Modelo
    class CropLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(26, 64, batch_first = True)
            self.fc = nn.Linear(64, n_classes)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[ : ,-1, :]
            return self.fc(out)
    
    #Criando variáveis de métricas
    all_preds_total = []
    all_true_total = []
    best_f1 = 0

    #Carregando o modelo TREINADO
    model = CropLSTM().to(device)
    model.load_state_dict(
            torch.load('model_2019_fr.pt')
    )
    
    #Reshape
    X_train = X_train_np.reshape(-1, 12, 26)
    X_test = X_test_np.reshape(-1, 12, 26)

    #Montando o tensor
    X_train = torch.tensor(X_train, dtype = torch.float32)
    X_test = torch.tensor(X_test, dtype = torch.float32)
    y_train = torch.tensor(y_train, dtype = torch.long)
    y_test = torch.tensor(y_test, dtype = torch.long)

    train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size = 128,
            shuffle = True  
    )

    test_loader = DataLoader(
            TensorDataset(X_test, y_test),
            batch_size = 128,
            shuffle = False
    )        
    

    #Critérios de valoração, aprendizado e evolução
    criterion = torch.nn.CrossEntropyLoss(weight = pesos.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    #Variáveis Early Stopping
    best_loss = float('inf')
    patience = 10
    min_delta = 0.001
    counter = 0

    #Treinamento
    for epoch in range(200):
        model.train()
        total_loss = 0

        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            pred = model(Xb)
            loss = criterion(pred, yb)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        #Condicional early stopping
        epoch_loss = total_loss/len(train_loader)
        print(f'Epoch{epoch+1} Loss {epoch_loss:.4f}')
        if best_loss - epoch_loss > min_delta:
             best_loss = epoch_loss
             counter = 0
        else:
             counter += 1

        if counter >= patience:
            print(f'Early stopping at {epoch +1}')
            break 
    #Avaliação do modelo
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)

            pred = model(Xb)
            _, predicted = torch.max(pred, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_true.extend(yb.cpu().numpy())

        all_preds_total.extend(all_preds)
        all_true_total.extend(all_true)

        acc = accuracy_score(all_true, all_preds)
        f1_macro = f1_score(all_true, all_preds, average = 'macro')
        precision_macro = precision_score(all_true, all_preds, average = 'macro')
        
        report = classification_report(all_true, all_preds, target_names = class_names, zero_division = 0)

        print(report)
        print(f'\nAccuracy:{acc:.4f}')
        print(f'F1 Macro:{f1_macro:.4f}')
        print(f'Precision Macro: {precision_macro:.4f}')

        #exportando dados
        if f1_macro > best_f1:
            best_f1 = f1_macro
            #Salvando o modelo
            torch.save(model.state_dict(), 'model_ft_2019_fr_with_2019_cat.pt')

    #Salvar resultados para gráficos e tabelas
    np.save('ytrue_2019_fr_with_2019_cat.npy', np.array(all_true_total))
    np.save('ypred_2019_fr_with_2019_cat.npy', np.array(all_preds_total))
    np.save('epoch_loss_2019_fr_with_2019_cat.npy',np.array(epoch_loss))
    #Gerar .txt com resultados
    with open('resultado_2019_fr_with_2019_cat.txt', 'w') as f:
        f.write(f'Resultados\n')
        f.write(f'Média Final\n')
        f.write(f'Accuracy:{acc:.4f}\n')
        f.write(f'F1 Macro:{f1_macro:.4f}\n')
        f.write(f'Precision:{precision_macro:.4f}\n')
        f.write(f'Relatório final de classificação')
        f.write(report)

if __name__ == '__main__':
    main()








