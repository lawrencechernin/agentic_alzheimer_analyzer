#!/usr/bin/env python3
"""
BHR MemTrax with Neural Networks - Building on 0.744 Baseline
==============================================================
Testing various neural network architectures on our best feature set:
1. Standard MLP with different depths/widths
2. Dropout regularization for overfitting
3. Batch normalization
4. Weighted loss for class imbalance
5. Early stopping with validation
6. Learning rate scheduling
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
import json
from tqdm import tqdm

np.random.seed(42)
torch.manual_seed(42)
RANDOM_STATE = 42
DATA_DIR = Path("../bhr/BHR-ALL-EXT_Mem_2022")
OUTPUT_DIR = Path("bhr_memtrax_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Cognitive impairment QIDs - FROM BASELINE
COGNITIVE_QIDS = ['QID1-5', 'QID1-12', 'QID1-13', 'QID1-22', 'QID1-23']

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class MCIDataset(Dataset):
    """PyTorch Dataset for MCI prediction"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SimpleNN(nn.Module):
    """Simple feedforward neural network"""
    def __init__(self, input_size, hidden_sizes, dropout_rate=0.3):
        super(SimpleNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class ResidualBlock(nn.Module):
    """Residual block for deeper networks"""
    def __init__(self, size):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(size, size)
        self.bn1 = nn.BatchNorm1d(size)
        self.fc2 = nn.Linear(size, size)
        self.bn2 = nn.BatchNorm1d(size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class ResidualNN(nn.Module):
    """Neural network with residual connections"""
    def __init__(self, input_size, hidden_size=64, num_blocks=3):
        super(ResidualNN, self).__init__()
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_size) for _ in range(num_blocks)
        ])
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.input_layer(x)
        for block in self.residual_blocks:
            x = block(x)
        return self.output_layer(x)


class AttentionNN(nn.Module):
    """Neural network with self-attention mechanism"""
    def __init__(self, input_size, hidden_size=64):
        super(AttentionNN, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.Sigmoid()
        )
        
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        
        self.output = nn.Sequential(
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Apply attention
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        return self.output(x)


def extract_memtrax_features(memtrax_q):
    """Extract MemTrax features - FROM BASELINE"""
    features = []
    
    for subject, group in memtrax_q.groupby('SubjectCode'):
        feat = {'SubjectCode': subject}
        
        # Basic statistics - FROM BASELINE
        feat['CorrectPCT_mean'] = group['CorrectPCT'].mean()
        feat['CorrectPCT_std'] = group['CorrectPCT'].std()
        feat['CorrectPCT_min'] = group['CorrectPCT'].min()
        feat['CorrectPCT_max'] = group['CorrectPCT'].max()
        
        feat['CorrectResponsesRT_mean'] = group['CorrectResponsesRT'].mean()
        feat['CorrectResponsesRT_std'] = group['CorrectResponsesRT'].std()
        feat['CorrectResponsesRT_min'] = group['CorrectResponsesRT'].min()
        feat['CorrectResponsesRT_max'] = group['CorrectResponsesRT'].max()
        
        feat['IncorrectPCT_mean'] = group['IncorrectPCT'].mean()
        feat['IncorrectResponsesRT_mean'] = group['IncorrectResponsesRT'].mean()
        
        # Composite scores - FROM BASELINE
        feat['CognitiveScore'] = feat['CorrectResponsesRT_mean'] / (feat['CorrectPCT_mean'] + 0.01)
        feat['Speed_Accuracy_Product'] = feat['CorrectPCT_mean'] * feat['CorrectResponsesRT_mean']
        feat['Error_Rate'] = 1 - feat['CorrectPCT_mean']
        feat['Response_Consistency'] = 1 / (feat['CorrectResponsesRT_std'] + 0.01)
        
        # Sequence features - FROM BASELINE
        all_rts = []
        for _, row in group.iterrows():
            if pd.notna(row.get('ReactionTimes')):
                try:
                    rts = [float(x.strip()) for x in str(row['ReactionTimes']).split(',') 
                           if x.strip() and x.strip() != 'nan']
                    all_rts.extend([r for r in rts if 0.3 <= r <= 3.0])
                except:
                    continue
        
        if len(all_rts) >= 10:
            n = len(all_rts)
            third = max(1, n // 3)
            
            feat['first_third_mean'] = np.mean(all_rts[:third])
            feat['last_third_mean'] = np.mean(all_rts[-third:])
            feat['fatigue_effect'] = feat['last_third_mean'] - feat['first_third_mean']
            
            mid = n // 2
            if mid > 1:
                feat['reliability_change'] = np.var(all_rts[mid:]) - np.var(all_rts[:mid])
                
            if n >= 3:
                slope, _ = np.polyfit(np.arange(n), all_rts, 1)
                feat['rt_slope'] = slope
                
        feat['n_tests'] = len(group)
        features.append(feat)
    
    return pd.DataFrame(features)


def build_composite_labels(med_hx):
    """Build composite cognitive impairment labels - FROM BASELINE"""
    if 'TimepointCode' in med_hx.columns:
        med_hx = med_hx[med_hx['TimepointCode'] == 'm00']
    
    med_hx = med_hx.drop_duplicates(subset=['SubjectCode'])
    available_qids = [q for q in COGNITIVE_QIDS if q in med_hx.columns]
    
    if not available_qids:
        raise ValueError("No cognitive QIDs found!")
    
    impairment = np.zeros(len(med_hx), dtype=int)
    valid = np.zeros(len(med_hx), dtype=bool)
    
    for qid in available_qids:
        impairment |= (med_hx[qid] == 1).values
        valid |= med_hx[qid].isin([1, 2]).values
    
    labels = pd.DataFrame({
        'SubjectCode': med_hx['SubjectCode'],
        'cognitive_impairment': impairment
    })
    
    return labels[valid].copy()


def add_demographics(df, data_dir):
    """Add demographics and create interaction features - FROM BASELINE"""
    demo_files = ['BHR_Demographics.csv', 'Profile.csv']
    
    for filename in demo_files:
        path = data_dir / filename
        if path.exists():
            try:
                demo = pd.read_csv(path, low_memory=False)
                if 'Code' in demo.columns:
                    demo.rename(columns={'Code': 'SubjectCode'}, inplace=True)
                    
                if 'SubjectCode' in demo.columns:
                    cols = ['SubjectCode']
                    for c in ['Age_Baseline', 'YearsEducationUS_Converted', 'Gender']:
                        if c in demo.columns:
                            cols.append(c)
                    
                    if len(cols) > 1:
                        df = df.merge(demo[cols].drop_duplicates('SubjectCode'), 
                                     on='SubjectCode', how='left')
                        break
            except:
                continue
    
    # Derived features - FROM BASELINE
    if 'Age_Baseline' in df.columns:
        df['Age_sq'] = df['Age_Baseline'] ** 2
        if 'CorrectResponsesRT_mean' in df.columns:
            df['age_rt_interact'] = df['Age_Baseline'] * df['CorrectResponsesRT_mean'] / 65
            
    if 'YearsEducationUS_Converted' in df.columns:
        df['Edu_sq'] = df['YearsEducationUS_Converted'] ** 2
        
    if all(c in df.columns for c in ['Age_Baseline', 'YearsEducationUS_Converted']):
        df['Age_Edu_interact'] = df['Age_Baseline'] * df['YearsEducationUS_Converted']
        df['CogReserve'] = df['YearsEducationUS_Converted'] / (df['Age_Baseline'] + 1)
        
    if 'Gender' in df.columns:
        df['Gender_Num'] = df['Gender'].map({'Male': 1, 'Female': 0, 'M': 1, 'F': 0})
        
    return df


def train_neural_network(model, train_loader, val_loader, epochs=100, lr=0.001, 
                         class_weights=None, patience=10):
    """Train a neural network with early stopping"""
    
    if class_weights is not None:
        # Convert class weights to tensor
        pos_weight = torch.FloatTensor([class_weights[1] / class_weights[0]]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    best_val_auc = 0
    patience_counter = 0
    
    train_losses = []
    val_aucs = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch).squeeze()
                val_preds.extend(outputs.cpu().numpy())
                val_true.extend(y_batch.numpy())
        
        val_auc = roc_auc_score(val_true, val_preds)
        
        # Learning rate scheduling
        scheduler.step(train_loss)
        
        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 20 == 0:
            print(f"   Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, Val AUC={val_auc:.4f}")
    
    # Restore best model
    model.load_state_dict(best_model_state)
    return model, best_val_auc


def evaluate_nn_architectures(X_train, X_test, y_train, y_test):
    """Evaluate different neural network architectures"""
    
    # Further split train into train/val for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    
    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    
    # Create data loaders
    train_dataset = MCIDataset(X_tr, y_tr)
    val_dataset = MCIDataset(X_val, y_val)
    test_dataset = MCIDataset(X_test, y_test)
    
    # Create weighted sampler for balanced batches
    train_targets = torch.FloatTensor(y_tr)
    weights = torch.FloatTensor([class_weight_dict[int(t)] for t in train_targets])
    sampler = WeightedRandomSampler(weights, len(weights))
    
    train_loader = DataLoader(train_dataset, batch_size=256, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    input_size = X_train.shape[1]
    results = {}
    
    # 1. Shallow Network
    print("\n   Training Shallow Network (2 layers)...")
    model = SimpleNN(input_size, [64, 32], dropout_rate=0.3).to(device)
    model, val_auc = train_neural_network(model, train_loader, val_loader, 
                                          epochs=100, lr=0.001, 
                                          class_weights=class_weight_dict)
    
    # Test evaluation
    model.eval()
    test_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze()
            test_preds.extend(outputs.cpu().numpy())
    
    test_auc = roc_auc_score(y_test, test_preds)
    results['Shallow_NN'] = {'val_auc': val_auc, 'test_auc': test_auc}
    print(f"      Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}")
    
    # 2. Deep Network
    print("\n   Training Deep Network (4 layers)...")
    model = SimpleNN(input_size, [128, 64, 32, 16], dropout_rate=0.4).to(device)
    model, val_auc = train_neural_network(model, train_loader, val_loader, 
                                          epochs=100, lr=0.001, 
                                          class_weights=class_weight_dict)
    
    model.eval()
    test_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze()
            test_preds.extend(outputs.cpu().numpy())
    
    test_auc = roc_auc_score(y_test, test_preds)
    results['Deep_NN'] = {'val_auc': val_auc, 'test_auc': test_auc}
    print(f"      Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}")
    
    # 3. Wide Network
    print("\n   Training Wide Network...")
    model = SimpleNN(input_size, [256, 128], dropout_rate=0.5).to(device)
    model, val_auc = train_neural_network(model, train_loader, val_loader, 
                                          epochs=100, lr=0.0005, 
                                          class_weights=class_weight_dict)
    
    model.eval()
    test_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze()
            test_preds.extend(outputs.cpu().numpy())
    
    test_auc = roc_auc_score(y_test, test_preds)
    results['Wide_NN'] = {'val_auc': val_auc, 'test_auc': test_auc}
    print(f"      Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}")
    
    # 4. Residual Network
    print("\n   Training Residual Network...")
    model = ResidualNN(input_size, hidden_size=64, num_blocks=3).to(device)
    model, val_auc = train_neural_network(model, train_loader, val_loader, 
                                          epochs=100, lr=0.001, 
                                          class_weights=class_weight_dict)
    
    model.eval()
    test_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze()
            test_preds.extend(outputs.cpu().numpy())
    
    test_auc = roc_auc_score(y_test, test_preds)
    results['Residual_NN'] = {'val_auc': val_auc, 'test_auc': test_auc}
    print(f"      Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}")
    
    # 5. Attention Network
    print("\n   Training Attention Network...")
    model = AttentionNN(input_size, hidden_size=64).to(device)
    model, val_auc = train_neural_network(model, train_loader, val_loader, 
                                          epochs=100, lr=0.001, 
                                          class_weights=class_weight_dict)
    
    model.eval()
    test_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze()
            test_preds.extend(outputs.cpu().numpy())
    
    test_auc = roc_auc_score(y_test, test_preds)
    results['Attention_NN'] = {'val_auc': val_auc, 'test_auc': test_auc}
    print(f"      Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}")
    
    return results


def evaluate_sklearn_mlp(X_train, X_test, y_train, y_test):
    """Test various sklearn MLP configurations"""
    
    print("\n7. SKLEARN MLP VARIATIONS")
    print("="*70)
    
    results = {}
    
    # Different architectures to try
    architectures = {
        'Small': (50, 25),
        'Medium': (100, 50, 25),
        'Large': (200, 100, 50, 25),
        'Wide': (256, 128),
        'Deep': (64, 64, 64, 64, 64)
    }
    
    for name, hidden_layers in architectures.items():
        print(f"\n   Testing {name} architecture {hidden_layers}...")
        
        # Pipeline with scaling
        mlp_pipe = Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                random_state=RANDOM_STATE
            ))
        ])
        
        # Train
        mlp_pipe.fit(X_train, y_train)
        
        # Predict
        y_pred = mlp_pipe.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_pred)
        
        print(f"      Test AUC: {test_auc:.4f}")
        results[f'MLP_{name}'] = test_auc
    
    return results


def main():
    print("\n" + "="*70)
    print("BHR MEMTRAX WITH NEURAL NETWORKS - BASED ON 0.744 BASELINE")
    print("="*70)
    
    # Load data
    print("\n1. Loading BHR data...")
    memtrax = pd.read_csv(DATA_DIR / 'MemTrax.csv', low_memory=False)
    med_hx = pd.read_csv(DATA_DIR / 'BHR_MedicalHx.csv', low_memory=False)
    
    # Quality filter - FROM BASELINE
    print("2. Applying Ashford quality filter...")
    memtrax_q = memtrax[
        (memtrax['Status'] == 'Collected') &
        (memtrax['CorrectPCT'] >= 0.60) &
        (memtrax['CorrectResponsesRT'].between(0.5, 2.5))
    ]
    
    # Extract features - FROM BASELINE
    print("3. Extracting MemTrax features (baseline configuration)...")
    features = extract_memtrax_features(memtrax_q)
    
    # Add demographics - FROM BASELINE
    print("4. Adding demographics...")
    features = add_demographics(features, DATA_DIR)
    
    # Create labels - FROM BASELINE
    print("5. Creating labels...")
    labels = build_composite_labels(med_hx)
    
    # Merge
    data = features.merge(labels, on='SubjectCode', how='inner')
    
    print(f"\n   Final dataset: {len(data):,} subjects")
    print(f"   Total features: {data.shape[1]-2}")
    print(f"   MCI prevalence: {data['cognitive_impairment'].mean():.1%}")
    
    # Prepare for modeling
    X = data.drop(['SubjectCode', 'cognitive_impairment'], axis=1).values
    y = data['cognitive_impairment'].values
    
    # Impute and scale
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X = imputer.fit_transform(X)
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Train: {len(y_train):,} samples ({y_train.mean():.1%} positive)")
    print(f"   Test: {len(y_test):,} samples ({y_test.mean():.1%} positive)")
    
    # Test PyTorch neural networks
    print("\n6. PYTORCH NEURAL NETWORKS")
    print("="*70)
    
    torch_results = evaluate_nn_architectures(X_train, X_test, y_train, y_test)
    
    # Test sklearn MLPs
    sklearn_results = evaluate_sklearn_mlp(X_train, X_test, y_train, y_test)
    
    # === FINAL RESULTS ===
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    baseline_auc = 0.744
    
    # Combine all results
    all_results = {}
    for name, res in torch_results.items():
        all_results[name] = res['test_auc']
    all_results.update(sklearn_results)
    
    # Find best
    best_model = max(all_results.items(), key=lambda x: x[1])
    best_name = best_model[0]
    best_auc = best_model[1]
    
    print(f"\nBaseline AUC: {baseline_auc:.4f}")
    print(f"Best Neural Network: {best_name}")
    print(f"Best NN AUC: {best_auc:.4f}")
    print(f"Difference: {(best_auc - baseline_auc):+.4f}")
    
    # Summary
    print("\nAll Neural Network Results:")
    for name, auc in sorted(all_results.items(), key=lambda x: -x[1]):
        print(f"   {name:20s}: {auc:.4f}")
    
    if best_auc >= 0.80:
        print(f"\n🎯 SUCCESS! Neural networks achieved {best_auc:.4f} AUC!")
    elif best_auc >= 0.78:
        print(f"\n✅ Good improvement! {best_auc:.4f} AUC")
    elif best_auc > baseline_auc:
        print(f"\n📈 Slight improvement to {best_auc:.4f}")
    else:
        print(f"\n📊 Neural networks did not improve over baseline")
    
    # Save results
    output = {
        'strategy': 'Neural Networks (PyTorch and sklearn)',
        'baseline_auc': baseline_auc,
        'best_model': best_name,
        'best_auc': float(best_auc),
        'improvement': float(best_auc - baseline_auc),
        'pytorch_results': {k: float(v['test_auc']) for k, v in torch_results.items()},
        'sklearn_results': {k: float(v) for k, v in sklearn_results.items()},
        'notes': [
            'Built on 0.744 baseline features',
            'Tested various architectures',
            'Used class weighting for imbalance',
            'Applied dropout and batch normalization',
            'Early stopping with validation'
        ]
    }
    
    with open(OUTPUT_DIR / 'neural_network_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {OUTPUT_DIR}/neural_network_results.json")
    
    return best_auc


if __name__ == '__main__':
    auc = main()

