# Pronto para o cenário 3 

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score
from sklearn.model_selection import KFold

#definindo pesos para o crossentropy
pesos = torch.tensor([
        3.0,  #Wheat
        1.0,  #Maize
        2.0,  #Sorghum
        1.0,  #Barley    
        2.0,  #Rye
        1.5,  #Oats    
        1.0,  #Grapes
        2.0,  #Rapeseed
        2.5,  #Sunflower
        2.0,  #Potato
        1.5,  #Pea
])
#Pesos de cada classe         
samples_per_class = {
            110:1000, # Wheat
            120:2000, #Maize
            140:1000, #Sorghum
            150:5000, #Barley
            160:1000, #Rye
            170:2500, #Oats
            330:5000, #Grapes
            435:1500, #Rapeseed
            438:1000, #Sunflower
            510:1000, #Potato
            770:1500 #Pea
}

def main():

    # Carregando os dados em um dataframe
    df = pd.read_csv("coco_full_2020.csv.gz")

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

    #Encoder linear fixo (elimina inconsistências entre datasets)
    linear_encoder = {k: i for i, k in enumerate(sorted(selected_classes.keys()))}

    # Filtrando somente as 11 classes que usaremos
    df = df[df["label"].isin(selected_classes.keys())]

    #Balanceando as classes
    dfs = []
    for label in selected_classes.keys():
        df_class = df[df['label'] == label]
    
        target_n = samples_per_class[label]

        n_samples = min(len(df_class), target_n)

        df_sampled = df_class.sample(n=n_samples, random_state=42)

        dfs.append(df_sampled)
    #Concatenção do dataframe determinado
    df = pd.concat(dfs, ignore_index=True)
    
    #Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Informando as features
    feature_cols = [c for c in df.columns if "mean" in c or "std" in c]
    X = df[feature_cols].values
    
    # Encoder
    y = df['label'].map(linear_encoder).values
    n_classes = len(linear_encoder)
    class_names = [selected_classes[k] for k in sorted(selected_classes.keys())]
    y = torch.tensor(y, dtype=torch.long)

    # Forçar GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Modelos
    class CropLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(26, 64, batch_first=True)
            self.fc = nn.Linear(64, n_classes)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            return self.fc(out)

    # Módulo do K-fold
    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    all_preds_total = []
    all_true_total = []
    accs = []
    f1s = []
    precisions = []
    best_acc = 0

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        # numpy
        X_train_np = X[train_idx]
        X_test_np = X[test_idx]

        #scaling
        scaler = StandardScaler()
        X_train_np = scaler.fit_transform(X_train_np)
        X_test_np = scaler.transform(X_test_np)

        #reshape
        X_train = X_train_np.reshape(-1,12,26)
        X_test = X_test_np.reshape(-1,12,26)

        #tensor
        X_train = torch.tensor(X_train, dtype = torch.float32)
        X_test = torch.tensor(X_test, dtype = torch.float32)


        # labels (já estão tensor)
        y_train = y[train_idx]
        y_test = y[test_idx]


        print(f"\n FOLD {fold+1} ")


        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=128,
            shuffle=True
        )

        test_loader = DataLoader(
            TensorDataset(X_test, y_test),
            batch_size=128,
            shuffle=False
        )

        # Resetando para um novo modelos
        model = CropLSTM().to(device)
        #Critério de valoração para correção
        criterion = torch.nn.CrossEntropyLoss(weight=pesos.to(device))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        #Variáveis early stopping
        best_loss = float('inf')
        patience = 10
        min_delta = 0.001
        counter = 0

        # Treinamento
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

            #Condicional early stoping
            epoch_loss = total_loss/len(train_loader)
            print(f'Epoch {epoch + 1} Loss {epoch_loss:.4f}')
            if best_loss - epoch_loss > min_delta:
                best_loss = epoch_loss
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                print(f'Early stopping na época {epoch + 1}')
                break

        # Avaliação dos modelos
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
        f1_macro = f1_score(all_true, all_preds, average="macro")
        precision_macro = precision_score(all_true, all_preds, average="macro")

        report = classification_report(all_true, all_preds, target_names = class_names, zero_division = 0)
        print(report)
        print(f"\nFold {fold+1} Accuracy: {acc:.4f}")
        print(f"Fold {fold+1} F1 Macro: {f1_macro:.4f}")
        print(f"Fold {fold+1} Precision Macro: {precision_macro:.4f}\n")

        accs.append(acc)
        f1s.append(f1_macro)
        precisions.append(precision_macro)

        # Exportando dados
        if acc > best_acc:
            best_acc = acc
            # Salvando o modelo
            torch.save(model.state_dict(), "model_2020_cat_cen3.pt")
            # Exportando o scaler
            joblib.dump(scaler, 'scaler_2020_cat_cen3.pkl')
            # Salvar encoder
            joblib.dump(linear_encoder, 'linear_encoder_2020_cat_cen3.pkl')
           
    # Resultados
    final_report = classification_report(all_true_total, all_preds_total, target_names = class_names, zero_division=0)
    mean_acc = np.mean(accs)
    mean_f1 = np.mean(f1s)
    mean_precision = np.mean(precisions)

    print("\nResultados finais")
    print(f"Mean Accuracy: {mean_acc:.4f}")
    print(f"Mean F1 Macro: {mean_f1:.4f}")
    print(f"Mean Precision: {mean_precision:.4f}")
    print('Relatório Final de Classificação')
    print(final_report)

    #Salvar resultados para gráficos
    np.save('ytrue_2020_cat_cen3.npy', np.array(all_true_total))
    np.save('ypred_2020_cat_cen3.npy', np.array(all_preds_total))

    # Criar .txt com resultados
    with open("resultado_2020_cat_cen3.txt", "w") as f:
        f.write("Resultados K-FOLD \n\n")

        for i in range(5):
            f.write(f"Fold {i+1} Accuracy: {accs[i]:.4f}\n")
            f.write(f"Fold {i+1} F1: {f1s[i]:.4f}\n")
            f.write(f"Fold {i+1} Precision: {precisions[i]:.4f}\n\n")

        f.write("Média Final\n")
        f.write(f"Mean Accuracy: {mean_acc:.4f}\n")
        f.write(f"Mean F1 Macro: {mean_f1:.4f}\n")
        f.write(f"Mean Precision: {mean_precision:.4f}\n")
        f.write(f'\nRelatório final de classificação\n\n')
        f.write(f'\nO scaling foi realizado a cada fold\n')
        f.write(final_report)
        

if __name__ == "__main__":
    main()
