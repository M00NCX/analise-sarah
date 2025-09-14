# Imports necessários
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin

# --- 1. CONFIGURAÇÃO INICIAL E DIRETÓRIOS ---
print("Criando diretórios para salvar os resultados...")
os.makedirs('graficos_modelos_cv_total', exist_ok=True)
os.makedirs('modelos_salvos_cv_total', exist_ok=True)

warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (12, 8)


# --- 2. FUNÇÕES DE PRÉ-PROCESSAMENTO ---

def msc(X):
    X = np.asarray(X)
    mean_spectrum = np.mean(X, axis=0)
    corrected_spectra = np.zeros_like(X)
    for i in range(X.shape[0]):
        slope, intercept = np.polyfit(mean_spectrum, X[i, :], 1)
        corrected_spectra[i, :] = (X[i, :] - intercept) / slope
    return corrected_spectra

def snv(X):
    X = np.asarray(X)
    return (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)

def savitzky_golay(X, window_size=11, poly_order=2, deriv_order=1):
    return savgol_filter(X, window_length=window_size, polyorder=poly_order, deriv=deriv_order, axis=1)

class OrthogonalCorrection(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=1):
        self.n_components = n_components

    def fit(self, X, y=None):
        X, y = np.asarray(X), np.asarray(y).ravel()
        self.w_ortho_, self.p_ortho_ = [], []
        X_corr = X.copy()
        for _ in range(self.n_components):
            pls = PLSRegression(n_components=1)
            pls.fit(X_corr, y)
            w, p = pls.x_weights_, pls.x_loadings_
            w_ortho = p - (np.dot(w.T, p) / np.dot(w.T, w)) * w
            t_ortho = np.dot(X_corr, w_ortho)
            p_ortho = np.dot(t_ortho.T, X_corr) / np.dot(t_ortho.T, t_ortho)
            X_corr -= np.dot(t_ortho, p_ortho)
            self.w_ortho_.append(w_ortho)
            self.p_ortho_.append(p_ortho)
        return self

    def transform(self, X, y=None):
        X_res = np.asarray(X).copy()
        for i in range(self.n_components):
            t_ortho = np.dot(X_res, self.w_ortho_[i])
            X_res -= np.dot(t_ortho, self.p_ortho_[i])
        return X_res

# --- 3. FUNÇÕES AUXILIARES (CARREGAMENTO E PLOTAGEM) ---

def load_data(filepath):
    df = pd.read_excel(filepath, engine='openpyxl')
    numeric_cols = [col for col in df.columns if str(col).replace('.', '', 1).isdigit()]
    metadata = df.drop(columns=numeric_cols)
    wavelengths = df[numeric_cols]
    return metadata, wavelengths

def save_cv_plot(y_true, y_pred_cv, atributo, filtro, modelo, file_path):
    """Gera e salva um gráfico comparando predições da validação cruzada."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        r2_cv = r2_score(y_true, y_pred_cv)
        rmse_cv = np.sqrt(mean_squared_error(y_true, y_pred_cv))

        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred_cv, alpha=0.7, edgecolors='k')
        min_val = min(y_true.min(), y_pred_cv.min())
        max_val = max(y_true.max(), y_pred_cv.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Linha Ideal (1:1)')
        plt.xlabel("Valores Reais")
        plt.ylabel("Valores Preditos (CV)")
        plt.title(f'Desempenho CV: {atributo}\nModelo: {modelo} | Filtro: {filtro}')
        stats_text = f'R²_CV = {r2_cv:.4f}\nRMSE_CV = {rmse_cv:.4f}'
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.5))
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(file_path, format='png', dpi=200, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"  Erro ao gerar gráfico para {modelo} com {filtro}: {e}")
        plt.close()

# --- 4. CARREGAMENTO E DEFINIÇÃO DOS PARÂMETROS ---

filepath = 'Data/raw/Fruto/Avaliacao_Maturacao_Palmer_e_Tommy_Fieldspec.xlsx'
metadata, wavelengths = load_data(filepath)
X = wavelengths.values
atributos = ['Firmness (N)', 'Dry Mass (%)', 'TSS (Brix)', 'TA (g/mL)', 'AA (mg/100g)','Weight (g)','Width (mm)','Length (mm)']

print(f'Dados carregados: {X.shape[0]} amostras, {X.shape[1]} comprimentos de onda\n')

todos_filtros = {
    'Raw': None, 'MSC': msc, 'SNV': snv,
    'SG_D1': lambda X: savitzky_golay(X, deriv_order=1),
    'SG_D2': lambda X: savitzky_golay(X, deriv_order=2),
    'SNV_SG_D1': lambda X: savitzky_golay(snv(X), deriv_order=1),
    'OSC_1': OrthogonalCorrection(n_components=1),
    'OSC_2': OrthogonalCorrection(n_components=2)
}

modelos = {
    'PLSR': {'estimator': PLSRegression(), 'params': {'n_components': [5, 10, 15, 20]}},
    'PCR': {'estimator': Pipeline([('pca', PCA()), ('regressor', LinearRegression())]), 'params': {'pca__n_components': [5, 10, 15, 20]}},
    'RFR': {'estimator': RandomForestRegressor(random_state=42), 'params': {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}},
    'SVR': {'estimator': SVR(), 'params': {'C': [1, 10, 100], 'gamma': ['scale', 'auto'], 'kernel': ['rbf']}}
}

# --- 5. PIPELINE DE MODELAGEM COM VALIDAÇÃO CRUZADA TOTAL ---

lista_resultados_finais = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for atributo in atributos:
    print(f'\n{"="*40}\nINICIANDO MODELAGEM PARA: {atributo}\n{"="*40}')
    
    # a. Preparar dados do atributo (remover NaNs)
    y_full = metadata[atributo].values
    valid_mask = ~np.isnan(y_full)
    X_clean, y_clean = X[valid_mask], y_full[valid_mask]
    
    if len(y_clean) < kf.get_n_splits():
        print(f"Aviso: Poucas amostras ({len(y_clean)}). Pulando atributo.\n")
        continue

    for nome_filtro, filtro_func in todos_filtros.items():
        print(f'\n--- Testando Filtro: {nome_filtro} ---')
        
        # b. Aplicar filtros independentes de y (se houver)
        X_processed = X_clean.copy()
        if filtro_func and not isinstance(filtro_func, OrthogonalCorrection):
            X_processed = filtro_func(X_processed)

        for nome_modelo, info_modelo in modelos.items():
            start_time = time.time()
            
            # c. Montar o pipeline para o GridSearchCV
            steps = [('scaler', StandardScaler())]
            est_temp = info_modelo['estimator']
            params_grid = {f'regressor__{key}': value for key, value in info_modelo['params'].items()}
            
            # Se o filtro for dependente de y (OSC), ele entra no pipeline
            if isinstance(filtro_func, OrthogonalCorrection):
                steps.insert(0, ('filtro_osc', filtro_func))
            
            pipeline = Pipeline(steps + [('regressor', est_temp)])
            
            # d. GridSearchCV para encontrar os melhores hiperparâmetros
            grid = GridSearchCV(pipeline, params_grid, cv=kf, scoring='r2', n_jobs=-1)
            grid.fit(X_processed, y_clean)
            
            melhor_modelo_pipeline = grid.best_estimator_
            
            # e. Avaliação final com cross_val_predict para plotagem e métrica robusta
            y_pred_cv = cross_val_predict(melhor_modelo_pipeline, X_processed, y_clean, cv=kf, n_jobs=-1)
            
            r2_final_cv = r2_score(y_clean, y_pred_cv)
            rmse_final_cv = np.sqrt(mean_squared_error(y_clean, y_pred_cv))
            
            end_time = time.time()
            print(f'  - Modelo: {nome_modelo:<10} | R²_CV: {r2_final_cv:.4f} | RMSE_CV: {rmse_final_cv:.4f} | Tempo: {end_time - start_time:.2f}s')

            # f. Salvar o gráfico da melhor combinação
            path_grafico = f'graficos_modelos_cv_total/{atributo}_{nome_modelo}_{nome_filtro}.png'
            save_cv_plot(y_clean, y_pred_cv, atributo, nome_filtro, nome_modelo, path_grafico)
            
            # g. Armazenar os resultados
            lista_resultados_finais.append({
                'Atributo': atributo, 'Modelo': nome_modelo, 'Filtro': nome_filtro,
                'R2_CV': r2_final_cv, 'RMSE_CV': rmse_final_cv,
                'Melhores_Params_Grid': str(grid.best_params_),
                'objeto_modelo': melhor_modelo_pipeline
            })

# --- 6. ANÁLISE E EXPORTAÇÃO DOS RESULTADOS ---
print('\n\n✅ Modelagem exaustiva concluída!')
df_resultados = pd.DataFrame(lista_resultados_finais)

# Salva os 5 melhores modelos (baseado no R² de CV)
print("\n--- Salvando os 5 melhores modelos por atributo ---")
for atributo in atributos:
    top5 = df_resultados[df_resultados['Atributo'] == atributo].sort_values(by='R2_CV', ascending=False).head(5)
    for i, row in top5.iterrows():
        # Adiciona o rank no nome do arquivo (1 a 5)
        rank = i + 1 if isinstance(i, int) else top5.index.get_loc(i) + 1
        nome_arquivo = f"modelos_salvos_cv_total/top_{rank}_{row['Atributo']}_{row['Modelo']}_{row['Filtro']}.joblib"
        joblib.dump(row['objeto_modelo'], nome_arquivo)
        print(f"  - Modelo salvo: {nome_arquivo} (R²_CV: {row['R2_CV']:.4f})")

# Exporta a tabela completa de resultados para Excel
df_export = df_resultados.drop(columns='objeto_modelo').sort_values(by=['Atributo', 'R2_CV'], ascending=[True, False])
df_export.to_excel('analise_completa_resultados_cv_total.xlsx', index=False)
print('\n✅ Tabela de resultados salva em "analise_completa_resultados_cv_total.xlsx"')

# Exibe os melhores resultados por atributo
print("\n--- Melhores Combinações por Atributo (baseado no R² da Validação Cruzada) ---")
melhores_resultados = df_export.groupby('Atributo').first().reset_index()
colunas_para_exibir = ['Atributo', 'Modelo', 'Filtro', 'R2_CV', 'RMSE_CV', 'Melhores_Params_Grid']
print(melhores_resultados[colunas_para_exibir].to_string(index=False))

