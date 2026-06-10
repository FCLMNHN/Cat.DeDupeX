import pandas as pd
import re
import csv
import os
import sys
import unicodedata 
from fuzzywuzzy import fuzz
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import requests 
import io 
from datetime import datetime 
from pathlib import Path 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QHeaderView, QLabel, QSplitter, QSizePolicy, QTextEdit, 
    QTableView, QAbstractItemView, QMessageBox, QDialog, QLineEdit, 
    QCheckBox, QGroupBox, QGridLayout, QComboBox, QProgressBar,
    QLayout # Ajout de QLayout pour la contrainte de taille
)
from PySide6.QtCore import Qt, QThread, Signal, QModelIndex
from PySide6.QtGui import QStandardItemModel, QStandardItem, QColor, QIntValidator 

load_dotenv()

# Configuration globale
# URL du catalogue
SOURCE_URL = "https://www.pndb.fr/api/explore/v2.1/catalog/exports/csv?delimiter=%3B&list_separator=%2C&quote_all=false&with_bom=true" 
# Clé d'API pour dépublier
API_KEY = os.getenv("API_KEY")
INPUT_DELIMITER = ";" 
OUTPUT_DELIMITER = "ǂ"
OUTPUT_FILENAME = "rapport_doublons_final.csv"

# Paramètres de persistance
CATALOG_CACHE_DIR = "catalogs_cache"
CATALOG_FILE_EXTENSION = ".csv.gz"

# Colonne contenant l'identifiant unique du jeu de données pour l'API
UID_COLUMN_NAME = 'datasetid' 

# Les colonnes à afficher dans le tableau du rapport
REPORT_COLUMNS = [
    'NIVEAU_DOUBLON', 'SCORE_SIMILARITE', 'CRITERE_DETECTION',
    'LIGNE_1_NUMERO', 'TITRE_LIGNE_1',
    'LIGNE_2_NUMERO', 'TITRE_LIGNE_2'
]

COLUMNS_TO_ANALYZE = [
    'default.references', 
    'default.title',      
    'default.description',
    'default.publisher',  
    'default.keyword',    
]
NUM_PROCESSES = os.cpu_count() if os.cpu_count() else 4 
BLOCK_KEY_LENGTH = 5 
MIN_LENGTH_FOR_DESC_WEIGHT = 20
# -----------------------------------

# ==============================================================================
# PARTIE 1 : logique de déduplication
# ==============================================================================


def clean_text(text: str, is_description: bool = False) -> str:
    """ Nettoie le texte (minuscules, retrait de ponctuation, standardisation des espaces). """
    if not isinstance(text, str):
        return ""
        
    text = unicodedata.normalize('NFKC', text)
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')
    text = re.sub(r'[\u200B-\u200F\uFEFF\u202A-\u202E\u00AD\u2060-\u2064]', '', text)

    if is_description:
        text = re.sub(r'<a\s+href=.*?>.*?</a>', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'<br\s*/?>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'lien vers les donn\w*es', '', text, flags=re.IGNORECASE)
        text = re.sub(r'fichier pr\w*sentant les donn\w*es d\w*origine et la m\w*thode de d\w*termination du budget du conservatoire du littoral', '', text, flags=re.IGNORECASE)

    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_newlines(text: str) -> str:
    """ Retire les retours à la ligne (LF et CR) d'une chaîne de caractères pour le rapport CSV. """
    if not isinstance(text, str):
        return ""
    return text.replace('\n', ' ').replace('\r', ' ').strip() 

def highlight_diff(text1: str, text2: str) -> str:
    """ Surligne les mots différents dans la LIGNE 2 par rapport à la LIGNE 1 (en HTML). """
    
    def tokenize(text):
        text = text.lower()
        text = re.sub(r'[\W_]+', ' ', text)
        return text.split()

    words1 = tokenize(text1)
    words2 = tokenize(text2)

    set1 = set(words1)
    set2 = set(words2)

    diff_words = set2.difference(set1)
    
    current_text2 = text2
    tokens2 = re.findall(r'(\w+)', text2.lower()) 
    
    if not tokens2:
        return text2
    
    output_html = ""
    last_end = 0
    
    for word in tokens2:
        try:
            match = re.search(r'\b' + re.escape(word) + r'\b', current_text2[last_end:], re.IGNORECASE)
            
            if not match:
                match = re.search(re.escape(word), current_text2[last_end:], re.IGNORECASE)
                if not match:
                    continue

            start = match.start() + last_end
            end = match.end() + last_end
            
            original_word_slice = current_text2[start:end]
            
            output_html += current_text2[last_end:start]
            
            if word in diff_words:
                output_html += f"<span style='background-color: yellow; font-weight: bold;'>{original_word_slice}</span>"
            else:
                output_html += original_word_slice

            last_end = end
        except Exception:
            break 
            
    output_html += current_text2[last_end:]
    
    return output_html

# Fonction pour le calcul de similarité TF-IDF (Étape 2)
def calculate_tfidf_similarity(text1: str, text2: str) -> float:
    """ 
    Calcule la similarité cosinus TF-IDF entre deux chaînes (corpus ad-hoc).
    """
    if not text1 or not text2:
        return 0.0
    
    corpus = [text1, text2]
    
    # Configuration du vectoriseur : utilise le même token_pattern que le nettoyage.
    vectorizer = TfidfVectorizer(token_pattern=r'\b\w+\b') 
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
        
        # Similarité Cosinus entre le document 1 (ligne 0) et le document 2 (ligne 1)
        similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0] * 100
        return float(similarity)
    except ValueError:
        # Cas où les documents sont trop courts ou vides après vectorisation.
        return 0.0


def compare_chunk(
    df_cleaned: pd.DataFrame, 
    indices_pairs: List[Tuple[int, int]], 
    enabled_rules: List[str],
    similarity_threshold_strong: int,
    similarity_threshold_probable: int
) -> List[Tuple[int, int, str, float]]:
    """ Fonction de travail pour le multiprocessing (détection des doublons). """
    local_doubles: List[Tuple[int, int, str, float]] = []
    
    MIN_SIGNIFICANT_LENGTH = 20
    
    for i, j in indices_pairs:
        row1 = df_cleaned.iloc[i]
        row2 = df_cleaned.iloc[j]
        
        # Règle : identité totale (100%)
        if 'TOTAL' in enabled_rules:
            
            title1_clean = row1.get('default.title_CLEAN', '')
            title2_clean = row2.get('default.title_CLEAN', '')
            desc1_clean = row1.get('default.description_CLEAN', '')
            desc2_clean = row2.get('default.description_CLEAN', '')

            is_title_match = (title1_clean == title2_clean)
            is_desc_match = (desc1_clean == desc2_clean)
            is_valid_entry = (title1_clean != "" and len(desc1_clean) >= MIN_SIGNIFICANT_LENGTH)
            
            if is_title_match and is_desc_match and is_valid_entry:
                local_doubles.append((i, j, 'Titre et Description', 100.0))
                continue
        
        # Règle : identité forte (> Seuil FORTE) - FuzzyWuzzy
        if 'FORTE' in enabled_rules:
            title1_clean = row1.get('default.title_CLEAN', '')
            title2_clean = row2.get('default.title_CLEAN', '')
            desc1_clean = row1.get('default.description_CLEAN', '')
            desc2_clean = row2.get('default.description_CLEAN', '')
            
            # Score 1: Titre (Ratio - plus strict)
            score_title = fuzz.ratio(title1_clean, title2_clean)

            # Score 2: Description (Token Set Ratio - plus tolérant à l'ordre)
            score_desc = fuzz.token_set_ratio(desc1_clean, desc2_clean)
            
            # Calcul du score total de similarité : Pondération 2:1 (Titre:Description) si la description est significative
            if len(desc1_clean) > MIN_LENGTH_FOR_DESC_WEIGHT and len(desc2_clean) > MIN_LENGTH_FOR_DESC_WEIGHT:
                # Pondération (Titre * 2 + Description * 1) / 3
                score_total = (2 * score_title + 1 * score_desc) / 3.0
                criteria = 'Titre (Ratio) [66%] + Description (Token Set) [33%]'
            else:
                # Si la description n'est pas significative, on se base uniquement sur le titre
                score_total = score_title
                criteria = 'Titre (Ratio) Seul'

            # Vérification du seuil fort
            if score_total >= similarity_threshold_strong:
                local_doubles.append((i, j, criteria, float(score_total)))
                continue

        # Règle: TF-IDF (Similarité Cosinus)
        if 'TFIDF' in enabled_rules:
            title1_clean = row1.get('default.title_CLEAN', '')
            title2_clean = row2.get('default.title_CLEAN', '')
            desc1_clean = row1.get('default.description_CLEAN', '')
            desc2_clean = row2.get('default.description_CLEAN', '')
            
            # Score 1: Titre (Similarité Cosinus TF-IDF)
            score_title_tfidf = calculate_tfidf_similarity(title1_clean, title2_clean)

            # Score 2: Description (Similarité Cosinus TF-IDF)
            score_desc_tfidf = calculate_tfidf_similarity(desc1_clean, desc2_clean)
            
            # Détermination de la significativité
            # MIN_LENGTH_FOR_DESC_WEIGHT DOIT ÊTRE DEFINI (par exemple 20)
            is_desc1_significant = len(desc1_clean) > MIN_LENGTH_FOR_DESC_WEIGHT
            is_desc2_significant = len(desc2_clean) > MIN_LENGTH_FOR_DESC_WEIGHT
            
            if is_desc1_significant and is_desc2_significant:
                # Cas 1: Les deux descriptions sont significatives. Pondération normale (2:1).
                score_total_tfidf = (2 * score_title_tfidf + 1 * score_desc_tfidf) / 3.0
                criteria_tfidf = 'Titre (TF-IDF Cosine) [66%] + Description (TF-IDF Cosine) [33%]'
            else:
                # Cas 2: Au moins une description est insignifiante.
                
                # Vérification de l'asymétrie
                if is_desc1_significant != is_desc2_significant:
                    # Cas 2A: Asymétrie (une longue, une courte/vide). Pénalisation FORCÉE.
                    # Le score du titre est abaissé par l'absence d'une description longue.
                    score_desc_penalized = 0 # Forcer le score de description à 0
                    score_total_tfidf = (2 * score_title_tfidf + 1 * score_desc_penalized) / 3.0
                    criteria_tfidf = 'Titre (TF-IDF Cosine) Seul (Penalisé: DESC Asymétrie)'
                else:
                    # Cas 2B: Les deux descriptions sont insignifiantes (courtes ou vides).
                    # Se baser uniquement sur le titre est justifiable.
                    score_total_tfidf = score_title_tfidf
                    criteria_tfidf = 'Titre (TF-IDF Cosine) Seul'

            # Vérification du seuil fort
            if score_total_tfidf >= similarity_threshold_strong:
                local_doubles.append((i, j, criteria_tfidf, float(score_total_tfidf)))
                continue

        # Règle : identité probable (Seuil PROBABLE)
        if 'PROBABLE' in enabled_rules:
            score_keyword = fuzz.token_set_ratio(row1.get('default.keyword_CLEAN', ''), row2.get('default.keyword_CLEAN', ''))
            if score_keyword >= similarity_threshold_probable:
                local_doubles.append((i, j, 'default.keyword (Token Set)', float(score_keyword)))
                continue
            
            
    return local_doubles

def find_duplicates_multiprocess(
    df: pd.DataFrame, 
    enabled_rules: List[str],
    similarity_threshold_strong: int,
    similarity_threshold_probable: int
) -> Tuple[pd.DataFrame, List[List[Tuple[int, int]]], int]:
    """ Gère la préparation des données et l'identification des paires pour le Worker. """
    
    # Blocage (Blocking/Tiling)
    # df doit contenir les colonnes *_CLEAN à ce stade
    if 'default.title_CLEAN' in df.columns:
        df['BLOCK_KEY'] = df['default.title_CLEAN'].str[:BLOCK_KEY_LENGTH].fillna('')
    else:
        # Mesure de sécurité si le nettoyage a échoué
        df['BLOCK_KEY'] = '' 

    all_pairs: List[Tuple[int, int]] = []
    
    for _, group in df.groupby('BLOCK_KEY'):
        if len(group) < 2:
            continue
            
        indices = group.index.tolist() 
        
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                all_pairs.append((indices[i], indices[j]))
    
    cols_to_keep = [f'{col}_CLEAN' for col in ['default.title', 'default.description', 'default.publisher', 'default.keyword']]
    # Inclure les colonnes _CLEAN dans le DF pour le multiprocessing
    df_for_parallel = df[[c for c in cols_to_keep if c in df.columns]].copy()
        
    chunk_size = len(all_pairs) // NUM_PROCESSES
    chunks = [all_pairs[i:i + chunk_size] for i in range(0, len(all_pairs), chunk_size)]
    chunks = [c for c in chunks if c] 
    
    return df_for_parallel, chunks, len(all_pairs)

def generate_csv_report(doubles: Dict[str, List[Tuple[int, int, str, float]]], df: pd.DataFrame, filename: str) -> None:
    """ Génère un rapport de doublons dans un fichier CSV structuré. """
    
    all_matches = []
    # Assure l'unicité par paire (i, j) en priorisant le niveau de doublon le plus "fort"
    unique_pairs = {} 

    # Parcourir les niveaux de force (Total > Forte > Probable) pour n'enregistrer que le meilleur score
    level_order = ['Identité Totale', 'Identité Forte', 'Identité Probable']
    
    for level in level_order:
        for i, j, criteria, score in doubles.get(level, []):
            pair_key = tuple(sorted((i, j)))
            
            # Si la paire n'a pas encore été enregistrée ou si le niveau actuel est plus fort
            if pair_key not in unique_pairs:
                unique_pairs[pair_key] = (level, i, j, criteria, score)
                
    for (i, j), (level, i_row, j_row, criteria, score) in unique_pairs.items():
            
            # Utilisation des colonnes originales pour le rapport CSV
            row1_title = remove_newlines(df.iloc[i_row].get('default.title', ''))
            row2_title = remove_newlines(df.iloc[j_row].get('default.title', ''))
            row1_ref = remove_newlines(df.iloc[i_row].get('default.references', ''))
            row2_ref = remove_newlines(df.iloc[j_row].get('default.references', ''))
            
            row1_desc = remove_newlines(df.iloc[i_row].get('default.description', ''))
            row2_desc = remove_newlines(df.iloc[j_row].get('default.description', ''))
            row1_pub = remove_newlines(df.iloc[i_row].get('default.publisher', ''))
            row2_pub = remove_newlines(df.iloc[j_row].get('default.publisher', ''))
            row1_kw = remove_newlines(df.iloc[i_row].get('default.keyword', ''))
            row2_kw = remove_newlines(df.iloc[j_row].get('default.keyword', ''))
            
            all_matches.append({
                'NIVEAU_DOUBLON': level,
                'LIGNE_1_INDEX_0': i_row,
                'LIGNE_2_INDEX_0': j_row,
                'LIGNE_1_NUMERO': i_row + 2, 
                'LIGNE_2_NUMERO': j_row + 2,
                'SCORE_SIMILARITE': f"{score:.2f}%",
                'CRITERE_DETECTION': criteria,
                'TITRE_LIGNE_1': row1_title,
                'TITRE_LIGNE_2': row2_title,
                'REFERENCE_LIGNE_1': row1_ref,
                'REFERENCE_LIGNE_2': row2_ref,
                'DESCRIPTION_LIGNE_1': row1_desc,
                'DESCRIPTION_LIGNE_2': row2_desc,
                'PUBLISHER_LIGNE_1': row1_pub,
                'PUBLISHER_LIGNE_2': row2_pub,
                'KEYWORDS_LIGNE_1': row1_kw,
                'KEYWORDS_LIGNE_2': row2_kw,
            })
            
    report_df = pd.DataFrame(all_matches).drop_duplicates(subset=['LIGNE_1_INDEX_0', 'LIGNE_2_INDEX_0'], keep='first')
    
    if not report_df.empty and 'TITRE_LIGNE_1' in report_df.columns:
        report_df = report_df.sort_values(by='TITRE_LIGNE_1', ascending=True)

    report_df.to_csv(filename, 
                     sep=OUTPUT_DELIMITER, 
                     index=False, 
                     encoding='utf-8', 
                     quoting=csv.QUOTE_ALL)
    
    return report_df.shape[0]

# ==============================================================================
# PARTIE 2 : WORKERS (Chargement, Analyse et Action API)
# ==============================================================================

class CatalogLoaderWorker(QThread):
    """ Worker pour télécharger et sauvegarder le catalogue en arrière-plan. """
    
    finished = Signal(bool, object, str) 
    progress = Signal(int, str) 

    def __init__(self, parent=None):
        super().__init__(parent)

    def run(self):
        """ Contient la logique bloquante du téléchargement et du traitement Pandas. """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            self.progress.emit(5, "Téléchargement du catalogue (1/3)...")
            headers = {'Authorization': f'Apikey {API_KEY}'}
            response = requests.get(SOURCE_URL, headers=headers, timeout=120) 
            response.raise_for_status() 

            csv_content = io.StringIO(response.content.decode('utf-8'))
            
            self.progress.emit(50, "Lecture et prétraitement des données (2/3)...")
            df = pd.read_csv(csv_content, 
                             sep=INPUT_DELIMITER, 
                             dtype=str, 
                             encoding='utf-8', 
                             on_bad_lines='skip',
                             engine='python',        
                             quotechar='"',          
                             doublequote=True) 
            
            # S'assurer que la colonne UID est présente, sinon l'action de dépublication échouera
            if UID_COLUMN_NAME not in df.columns:
                 print(f"ATTENTION: Colonne UID '{UID_COLUMN_NAME}' manquante. La dépublication ne fonctionnera pas.")

            df['CSV_LINE_NUMERO'] = df.index + 2 
            df = df.set_index('CSV_LINE_NUMERO')
            
            self.progress.emit(80, "Sauvegarde locale du catalogue (3/3)...")
            Path(CATALOG_CACHE_DIR).mkdir(exist_ok=True)
            filename_key = timestamp.replace(' ', '_').replace(':', '-') 
            filename = f"catalogue_{filename_key}{CATALOG_FILE_EXTENSION}"
            save_path = Path(CATALOG_CACHE_DIR) / filename
            
            df.to_csv(save_path, sep=INPUT_DELIMITER, compression='gzip', index=True, encoding='utf-8', quoting=csv.QUOTE_ALL)
            
            self.progress.emit(100, "Téléchargement terminé.")
            self.finished.emit(True, df, f"Catalogue du {timestamp} ({len(df)} lignes)")
            
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else 'N/A'
            msg = f"Erreur HTTP ({status_code}) lors du téléchargement. Vérifiez la clé API et l'URL: {e}"
            self.progress.emit(0, "Erreur.")
            self.finished.emit(False, None, msg)
        except requests.exceptions.RequestException as e:
            msg = f"Erreur de connexion lors du téléchargement : {e}"
            self.progress.emit(0, "Erreur.")
            self.finished.emit(False, None, msg)
        except Exception as e:
            msg = f"Erreur lors du traitement des données téléchargées : {e}"
            self.progress.emit(0, "Erreur.")
            self.finished.emit(False, None, msg)


class DeduplicationWorker(QThread):
    """ Worker pour exécuter l'analyse de déduplication en arrière-plan. """
    
    finished = Signal(bool, str) 
    progress = Signal(int, str) 

    def __init__(self, df: pd.DataFrame, settings: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.df = df 
        self.settings = settings

    def run(self):
        """ Contient la logique bloquante de l'analyse de déduplication. """
        try:
            # df est censé contenir les colonnes *_CLEAN à ce stade
            df_copy = self.df.copy() 
            for col in COLUMNS_TO_ANALYZE:
                if col not in df_copy.columns:
                    df_copy[col] = ''
            df_reset = df_copy.reset_index(drop=True)

            self.progress.emit(5, "Préparation et découpage des données (1/3)...")
            
            df_for_parallel, chunks, total_pairs = find_duplicates_multiprocess(
                df_reset, 
                self.settings['enabled_rules'],
                self.settings['similarity_threshold_strong'],
                self.settings['similarity_threshold_probable']
            )

            total_chunks = len(chunks)
            if total_chunks == 0:
                self.progress.emit(100, "Aucune paire à comparer. Analyse terminée.")
                self.finished.emit(True, "Analyse terminée, 0 paires à comparer.")
                return

            self.progress.emit(10, f"Démarrage de l'analyse parallèle sur {total_chunks} blocs (2/3)...")
            
            final_doubles = {'Identité Totale': [], 'Identité Forte': [], 'Identité Probable': []}
            
            common_args = (self.settings['enabled_rules'], self.settings['similarity_threshold_strong'], self.settings['similarity_threshold_probable'])
            
            with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
                futures = [executor.submit(compare_chunk, df_for_parallel, chunk, *common_args) for chunk in chunks]
                
                for i, future in enumerate(as_completed(futures)):
                    local_doubles = future.result()
                    
                    for i_pair, j_pair, criteria, score in local_doubles:
                        # Logique de détermination du niveau de mise à jour (Étape 6)
                        if criteria == 'Titre et Description':
                            level = 'Identité Totale'
                        # Les critères des règles FORTE (FuzzyWuzzy) et TFIDF contiennent "Titre (Ratio)" ou "TF-IDF Cosine"
                        elif 'Titre (Ratio)' in criteria or 'TF-IDF Cosine' in criteria: 
                            level = 'Identité Forte'
                        # Le critère de la règle PROBABLE contient "Token Set" (pour les mots-clés)
                        elif 'Token Set' in criteria:
                            level = 'Identité Probable'
                        else:
                            level = 'Identité Probable' 
                            
                        final_doubles[level].append((i_pair, j_pair, criteria, score))
                    
                    percent = 10 + int(80 * (i + 1) / total_chunks)
                    self.progress.emit(percent, f"Analyse des doublons ({i+1}/{total_chunks} blocs traités)")

            self.progress.emit(90, "Génération du rapport CSV final (3/3)...")
            
            generate_csv_report(final_doubles, df_reset, OUTPUT_FILENAME)
            
            self.progress.emit(100, "Analyse terminée.")
            self.finished.emit(True, f"Analyse terminée avec succès.")
            
        except Exception as e:
            error_msg = f"Une erreur inattendue s'est produite lors de l'analyse : {e}"
            self.progress.emit(0, "Erreur.")
            self.finished.emit(False, error_msg)


class ApiActionWorker(QThread):
    """ Worker pour exécuter des actions API (comme la dépublication) en arrière-plan. """
    
    # Signal: (success, message, line_num)
    finished = Signal(bool, str, int)  

    def __init__(self, datasetid: str, line_num: int, api_key: str, parent=None):
        super().__init__(parent)
        self.datasetid = datasetid
        self.line_num = line_num
        self.api_key = api_key
        self.base_url = "https://www.pndb.fr/api"

    def run(self):
        """ Exécute l'opération de dépublication en deux étapes : obtenir l'UID, puis dépublier. """
        try:
            # 1. Obtenir le dataset_uid (l'identifiant interne pour l'API automation)
            # Utilisation de datasetid dans l'URL de l'API EXPLORE comme fourni par l'utilisateur
            explore_url = f"{self.base_url}/explore/v2.1/catalog/datasets/{self.datasetid}?timezone=UTC&include_links=false&include_app_metas=false"
            
            response = requests.get(explore_url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Le dataset_uid pour l'API automation est l'ID interne ('dataset_uid') ou parfois l'id racine.
            # CORRECTION APPLIQUÉE ICI : Utiliser 'dataset_uid' au lieu de 'id'
            dataset_uid = data.get('dataset_uid') 
            
            if not dataset_uid:
                # Si 'dataset_uid' n'est pas trouvé, vérifiez la clé 'id' comme alternative.
                dataset_uid = data.get('id')
                
            if not dataset_uid:
                self.finished.emit(False, f"Erreur L{self.line_num}: ID automation ('dataset_uid' ou 'id') non trouvé dans la réponse pour datasetid '{self.datasetid}'.", self.line_num)
                return

            # 2. Envoyer la requête de dépublication (POST)
            unpublish_url = f"{self.base_url}/automation/v1.0/datasets/{dataset_uid}/unpublish/"
            headers = {'Authorization': f'Apikey {self.api_key}'}
            
            # L'API automation/v1.0 nécessite une requête POST
            response = requests.post(unpublish_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Vérifier la réponse de dépublication (le statut 204 No Content est souvent attendu)
            if response.status_code in [200, 204]:
                msg = f"✅ L{self.line_num}: Dépublication réussie pour '{dataset_uid}' (datasetid: {self.datasetid})."
                self.finished.emit(True, msg, self.line_num)
            else:
                error_msg = response.text
                msg = f"❌ L{self.line_num}: Échec de la dépublication (Statut {response.status_code}). Réponse: {error_msg[:100]}..."
                self.finished.emit(False, msg, self.line_num)

        except requests.exceptions.RequestException as e:
            msg = f"❌ L{self.line_num}: Erreur réseau/API lors de l'opération: {e}"
            self.finished.emit(False, msg, self.line_num)
        except Exception as e:
            msg = f"❌ L{self.line_num}: Erreur inattendue: {e}"
            self.finished.emit(False, msg, self.line_num)


# ==============================================================================
# PARTIE 3 : DIALOGUE DE CONFIGURATION
# ==============================================================================

class SettingsDialog(QDialog):
    
    def __init__(self, current_settings: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚙️ Paramètres de Déduplication")
        self.current_settings = current_settings
        self.new_settings = current_settings.copy()
        
        self._setup_ui()
        # Contrainte de taille après la mise en place du layout
        self.layout().setSizeConstraint(QLayout.SetFixedSize) 
    
    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        
        rules_group = QGroupBox("Règles de Détection (Cochez pour Activer)")
        rules_layout = QVBoxLayout()
        
        self.checkboxes = {}
        
        for rule_name in ['TOTAL', 'FORTE', 'PROBABLE', 'TFIDF']: 
            cb = QCheckBox(f"Règle '{rule_name}'")
            if rule_name in self.current_settings['enabled_rules']:
                cb.setChecked(True)
            self.checkboxes[rule_name] = cb
            rules_layout.addWidget(cb)
            
        rules_group.setLayout(rules_layout)
        main_layout.addWidget(rules_group)
        
        threshold_group = QGroupBox("Seuils de Similarité (Score en %)")
        threshold_layout = QGridLayout()
        
        threshold_layout.addWidget(QLabel("Seuil 'FORTE' (Titre Ratio):"), 0, 0)
        self.strong_threshold_input = QLineEdit(str(self.current_settings['similarity_threshold_strong']))
        self.strong_threshold_input.setValidator(QIntValidator(0, 100, self))
        threshold_layout.addWidget(self.strong_threshold_input, 0, 1)
        
        threshold_layout.addWidget(QLabel("Seuil 'PROBABLE' (Token Set):"), 1, 0)
        self.probable_threshold_input = QLineEdit(str(self.current_settings['similarity_threshold_probable']))
        self.probable_threshold_input.setValidator(QIntValidator(0, 100, self))
        threshold_layout.addWidget(self.probable_threshold_input, 1, 1)
        
        threshold_group.setLayout(threshold_layout)
        main_layout.addWidget(threshold_group)
        
        button_layout = QHBoxLayout()
        ok_button = QPushButton("Appliquer")
        ok_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Annuler")
        cancel_button.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        
        main_layout.addLayout(button_layout)
        
    def get_settings(self) -> Dict[str, Any]:
        """ Retourne les paramètres validés. """
        
        new_enabled_rules = []
        # La boucle parcourt 'TOTAL', 'FORTE', 'PROBABLE', 'TFIDF'
        for rule_name, checkbox in self.checkboxes.items(): 
            if checkbox.isChecked():
                new_enabled_rules.append(rule_name)
        
        try:
            strong_t = int(self.strong_threshold_input.text())
        except ValueError:
            strong_t = self.current_settings['similarity_threshold_strong'] 
            
        try:
            probable_t = int(self.probable_threshold_input.text())
        except ValueError:
            probable_t = self.current_settings['similarity_threshold_probable']

        self.new_settings['enabled_rules'] = new_enabled_rules
        self.new_settings['similarity_threshold_strong'] = max(0, min(100, strong_t))
        self.new_settings['similarity_threshold_probable'] = max(0, min(100, probable_t))
        
        return self.new_settings


# ==============================================================================
# PARTIE 4 : INTERFACE GRAPHIQUE PRINCIPALE
# ==============================================================================

class MainWindow(QMainWindow):
    """ Fenêtre principale de l'application de Data Quality. """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Outil de Data Quality - Déduplication")
        self.setGeometry(0, 0, 1400, 800)
        
        self.settings: Dict[str, Any] = {
            'enabled_rules': ['TOTAL', 'FORTE', 'PROBABLE', 'TFIDF'], 
            'similarity_threshold_strong': 99,
            'similarity_threshold_probable': 80
        }
        
        self.catalogs: Dict[str, pd.DataFrame] = {} 
        self.current_catalog_key: Optional[str] = None
        self.report_df: pd.DataFrame = pd.DataFrame()
        
        self.deduplication_worker = None 
        self.catalog_loader_worker = None 
        self.api_action_worker = None 
        
        # Variables pour stocker les IDs des lignes actuellement affichées
        self.current_datasetid_1: Optional[str] = None
        self.current_datasetid_2: Optional[str] = None
        self.current_line_num_1: Optional[int] = None
        self.current_line_num_2: Optional[int] = None

        self._setup_ui()
        
        self._load_cached_catalogs()
        self._load_report_on_startup() 
        
    
    # S'assurer que les colonnes *_CLEAN existent
    def _ensure_cleaned_columns_exist(self, df: pd.DataFrame) -> None:
        """ 
        Génère les colonnes de données nettoyées (*_CLEAN) si elles sont manquantes.
        Ceci est essentiel pour que la vue de comparaison fonctionne, même si l'analyse
        de déduplication n'a pas été lancée.
        """
        
        # Colonnes pour nettoyage standard
        for col in ['default.title', 'default.publisher', 'default.keyword']:
            clean_col = f'{col}_CLEAN'
            if col in df.columns and clean_col not in df.columns:
                df[clean_col] = df[col].apply(clean_text)
                
        # Colonne pour nettoyage spécifique (description)
        desc_col = 'default.description'
        clean_desc_col = 'default.description_CLEAN'
        if desc_col in df.columns and clean_desc_col not in df.columns:
            df[clean_desc_col] = df[desc_col].apply(lambda x: clean_text(x, is_description=True))

    def _load_cached_catalogs(self):
        cache_path = Path(CATALOG_CACHE_DIR)
        if not cache_path.exists():
            return
            
        loaded_count = 0
        catalog_files = sorted(list(cache_path.glob(f"catalogue_*{CATALOG_FILE_EXTENSION}")), reverse=True)
        
        self.catalog_selector.clear()
        
        for file_path in catalog_files:
            try:
                base_name = file_path.stem.replace('catalogue_', '')
                match = re.search(r'(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})', base_name)
                
                if match:
                    date_part = match.group(1)
                    time_part = f"{match.group(2)}:{match.group(3)}:{match.group(4)}"
                    key = f"{date_part} {time_part}"
                else:
                    key = file_path.stem 
                
                df = pd.read_csv(
                    file_path, 
                    sep=INPUT_DELIMITER, 
                    compression='gzip',
                    dtype=str, 
                    encoding='utf-8',
                    index_col='CSV_LINE_NUMERO', 
                    engine='python' 
                )
                
                # Assurer les colonnes de nettoyage au chargement du cache
                self._ensure_cleaned_columns_exist(df)
                
                self.catalogs[key] = df
                self.catalog_selector.addItem(key)
                loaded_count += 1
                
            except Exception:
                pass
                
        if loaded_count > 0:
            self.catalog_selector.setCurrentIndex(0) 
            self.catalog_selector.setEnabled(True)
            self.run_button.setEnabled(True)
            self.current_catalog_key = self.catalog_selector.currentText()
            self.status_label.setText(f"Statut: ✅ {loaded_count} catalogues précédemment chargés trouvés. Actif: '{self.current_catalog_key}'.")
        else:
            self.catalog_selector.setEnabled(False)
            self.run_button.setEnabled(False)


    def get_active_dataframe(self) -> pd.DataFrame:
        """ Retourne le DataFrame actif ou un DataFrame vide. """
        if self.current_catalog_key and self.current_catalog_key in self.catalogs:
            return self.catalogs[self.current_catalog_key]
        return pd.DataFrame()
        
    def open_settings_dialog(self):
        dialog = SettingsDialog(self.settings, self)
        
        if dialog.exec():
            new_settings = dialog.get_settings()
            self.settings = new_settings
            self.status_label.setText("Statut: Paramètres mis à jour. (Prêt à relancer l'analyse)")
            
    def update_progress_bar(self, value: int, text: str):
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"{text} %p%")
        self.progress_bar.setVisible(value > 0 and value < 100)
            
    def _setup_ui(self):
        """ Configuration de l'interface utilisateur. """
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)

        # 1. Contrôles (Haut)
        control_layout = QHBoxLayout()
        
        self.download_button = QPushButton("⬇️ Télécharger le Catalogue")
        self.download_button.clicked.connect(self.start_catalog_download)
        
        self.catalog_selector = QComboBox()
        self.catalog_selector.setMinimumWidth(300)
        self.catalog_selector.setToolTip("Sélectionnez la version du catalogue à analyser (y compris les versions persistantes).")
        self.catalog_selector.currentIndexChanged.connect(self.update_active_catalog)
        self.catalog_selector.setEnabled(False) 
        
        self.run_button = QPushButton("🚀 Lancer l'analyse de déduplication")
        self.run_button.clicked.connect(self.start_deduplication_analysis)
        self.run_button.setEnabled(False) 
        
        self.ground_truth_button = QPushButton("📊 Générer le ground truth")
        self.ground_truth_button.clicked.connect(self.generate_ground_truth)
        self.ground_truth_button.setToolTip("Exporte les lignes cochées dans un fichier CSV dédié.")

        self.settings_button = QPushButton("⚙️ Paramètres")
        self.settings_button.clicked.connect(self.open_settings_dialog)
        
        self.status_label = QLabel("Statut: Prêt. Aucun catalogue chargé.")
        
        self.progress_bar = QProgressBar() 
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setVisible(False)
        
        control_layout.addWidget(self.download_button)
        control_layout.addWidget(self.catalog_selector)
        control_layout.addWidget(self.run_button)
        control_layout.addWidget(self.settings_button)
        control_layout.addStretch()
        control_layout.addWidget(self.status_label)
        control_layout.addWidget(self.progress_bar) 
        control_layout.addWidget(self.ground_truth_button)
        main_layout.addLayout(control_layout)
        
        # 2. Séparateur (Milieu) : Rapport et Comparaison
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # A. Vue du Rapport (Tableau)
        report_group = QWidget()
        report_layout = QVBoxLayout(report_group)
        report_layout.addWidget(QLabel("Rapport de doublons trouvés:"))
        
        self.report_table = QTableView()
        self.report_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.report_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        
        self.report_model = QStandardItemModel()
        self.report_table.setModel(self.report_model)
        
        selection_model = self.report_table.selectionModel()
        if selection_model:
            selection_model.currentChanged.connect(self.show_comparison_view)
        
        MIN_HEIGHT = 150 # Estimer une hauteur raisonnable en pixels

        self.report_table.setMinimumHeight(MIN_HEIGHT)
        
        report_layout.addWidget(self.report_table)
        splitter.addWidget(report_group)
        
        # B. Vue de Comparaison (Bas)
        comparison_group = QWidget()
        comparison_layout = QVBoxLayout(comparison_group)
        comparison_layout.setContentsMargins(0, 0, 0, 0)

        # Zone pour les boutons et le titre de comparaison
        header_widget = QWidget()
        header_layout = QGridLayout(header_widget)
        header_layout.setContentsMargins(5, 5, 5, 5) 
        header_layout.setSpacing(5)
        
        self.comparison_title_label = QLabel("Vue de Comparaison (Textes Nettoyés):")
        self.comparison_title_label.setStyleSheet("font-weight: bold;")
        # Ajout du titre sur toute la largeur (colonnes 0, 1, 2)
        header_layout.addWidget(self.comparison_title_label, 0, 0, 1, 3) 

        # Boutons de dépublication
        self.unpublish_btn_1 = QPushButton("Dépublier LIGNE 1")
        self.unpublish_btn_2 = QPushButton("Dépublier LIGNE 2")
        self.unpublish_btn_1.setToolTip("Dépublie le jeu de données correspondant à la LIGNE 1 via l'API Automation PNDB.")
        self.unpublish_btn_2.setToolTip("Dépublie le jeu de données correspondant à la LIGNE 2 via l'API Automation PNDB.")
        self.unpublish_btn_1.setEnabled(False) 
        self.unpublish_btn_2.setEnabled(False) 
        
        # LIGNE 1: Colonne 0, Alignement à gauche (collé à gauche)
        header_layout.addWidget(self.unpublish_btn_1, 1, 0, Qt.AlignLeft)
        
        # ÉTIREMENT: Colonne 1, force le bouton 2 à droite
        header_layout.setColumnStretch(1, 1) 
        
        # LIGNE 2: Colonne 2, Alignement à droite (collé à droite)
        header_layout.addWidget(self.unpublish_btn_2, 1, 2, Qt.AlignRight) 
        
        comparison_layout.addWidget(header_widget)
        
        self.comparison_text = QTextEdit()
        self.comparison_text.setReadOnly(True)
        self.comparison_text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        comparison_layout.addWidget(self.comparison_text)
        splitter.addWidget(comparison_group)
        

        splitter.setSizes([300, 500])
        main_layout.addWidget(splitter)

        self.setCentralWidget(main_widget)
        
        # Connexion des nouveaux boutons
        self.unpublish_btn_1.clicked.connect(lambda: self.unpublish_button_clicked(1))
        self.unpublish_btn_2.clicked.connect(lambda: self.unpublish_button_clicked(2))
        
    def _load_report_on_startup(self):
        if os.path.exists(OUTPUT_FILENAME):
            self.load_report_into_table(OUTPUT_FILENAME)
        else:
            if not self.catalogs:
                self.status_label.setText("Statut: Prêt. Aucun catalogue chargé. Lancez un téléchargement.")
            
    def start_catalog_download(self):
        if self.catalog_loader_worker and self.catalog_loader_worker.isRunning():
            QMessageBox.warning(self, "Téléchargement en cours", "Un téléchargement est déjà en cours. Veuillez patienter.")
            return

        self.catalog_loader_worker = CatalogLoaderWorker()
        self.catalog_loader_worker.finished.connect(self.catalog_loaded_result)
        self.catalog_loader_worker.progress.connect(self.update_progress_bar) 
        
        self.status_label.setText("Statut: ⬇️ Téléchargement du catalogue depuis l'API...")
        
        self.download_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.settings_button.setEnabled(False) 
        self.catalog_selector.setEnabled(False) 
        
        self.catalog_loader_worker.start() 

    def catalog_loaded_result(self, success: bool, result: Optional[pd.DataFrame], message: str):
        self.download_button.setEnabled(True)
        self.settings_button.setEnabled(True) 
        self.update_progress_bar(0, "") 

        if success and result is not None:
            match = re.search(r'du (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', message)
            new_key = match.group(1) if match else message.split('(')[0].strip()
            
            # Assurer les colonnes de nettoyage après un nouveau téléchargement
            self._ensure_cleaned_columns_exist(result)
            
            self.catalogs[new_key] = result
            
            self._load_cached_catalogs()
            
            self.catalog_selector.setEnabled(True)
            self.run_button.setEnabled(True)
            self.status_label.setText(f"Statut: ✅ {message}. Catalogue actif: '{new_key}'. (Sauvegardé localement)")
            
        else:
            self.status_label.setText(f"Statut: ❌ Échec du téléchargement - {message}")
            QMessageBox.critical(self, "Erreur de Téléchargement", message)
            
            if not self.catalogs:
                self.run_button.setEnabled(False)
                self.catalog_selector.setEnabled(False)
            else:
                self.run_button.setEnabled(True)
                self.catalog_selector.setEnabled(True)
            
    def update_active_catalog(self):
        new_key = self.catalog_selector.currentText()
        if new_key and new_key in self.catalogs:
            self.current_catalog_key = new_key
            
            # S'assurer que le DF actif a les colonnes nettoyées après un changement
            active_df = self.get_active_dataframe()
            if not active_df.empty:
                self._ensure_cleaned_columns_exist(active_df)
            
            self.status_label.setText(f"Statut: Catalogue actif: '{new_key}'. Prêt.")
            self.load_report_into_table(OUTPUT_FILENAME) 
            self.run_button.setEnabled(True)
        else:
            self.current_catalog_key = None
            self.report_model.clear()
            self.status_label.setText("Statut: Veuillez télécharger ou sélectionner un catalogue.")
            self.run_button.setEnabled(False)

    def load_report_into_table(self, report_filename: str):
        active_df = self.get_active_dataframe()
        total_lines = len(active_df) if not active_df.empty else 0
        
        self.report_model.clear()
        
        try:
            self.report_df = pd.read_csv(
                report_filename, 
                sep=OUTPUT_DELIMITER, 
                dtype=str, 
                encoding='utf-8',
                quotechar='"',
                engine='python' 
            )
            
            display_df = self.report_df[REPORT_COLUMNS]

            self.report_model.setHorizontalHeaderLabels(display_df.columns.tolist())
            
            for index, row in display_df.iterrows():
                items = []
                for i, item in enumerate(row):
                    std_item = QStandardItem(str(item))
                    
                    if i == 0:
                        std_item.setCheckable(True)
                        std_item.setCheckState(Qt.Unchecked)
                    
                    if row['NIVEAU_DOUBLON'] == 'Identité Totale':
                        std_item.setBackground(QColor(255, 192, 192))
                    elif row['NIVEAU_DOUBLON'] == 'Identité Forte':
                        std_item.setBackground(QColor(255, 255, 192))
                    
                    items.append(std_item)
                self.report_model.appendRow(items)

            self.report_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            
            if not self.report_df.empty and not active_df.empty:
                lignes_doublons_uniques = pd.concat([
                    self.report_df['LIGNE_1_NUMERO'].astype(str), 
                    self.report_df['LIGNE_2_NUMERO'].astype(str)
                ]).nunique()
                
                self.status_label.setText(f"Statut: Catalogue actif: '{self.current_catalog_key}'. Rapport chargé. {len(self.report_df)} paires trouvées. "
                                        f"**Lignes uniques impliquées : {lignes_doublons_uniques} / {total_lines}**.")
            elif not active_df.empty:
                self.status_label.setText(f"Statut: Catalogue actif: '{self.current_catalog_key}'. 0 paires trouvées dans le rapport.")

        except Exception:
            if not active_df.empty:
                 self.status_label.setText(f"Statut: Catalogue actif: '{self.current_catalog_key}'. Erreur lors du chargement du rapport. Lancez l'analyse pour le générer.")
            else:
                 self.status_label.setText("Statut: Erreur lors du chargement du rapport. Aucun catalogue actif.")


    def start_deduplication_analysis(self):
        df_active = self.get_active_dataframe()
        
        if df_active.empty:
            QMessageBox.warning(self, "Erreur", "Veuillez d'abord télécharger ou sélectionner une version du catalogue.")
            return

        if not self.settings['enabled_rules']:
            QMessageBox.warning(self, "Configuration Invalide", "Au moins une règle de détection doit être sélectionnée dans les paramètres.")
            return

        if self.deduplication_worker and self.deduplication_worker.isRunning():
            QMessageBox.warning(self, "Analyse en cours", "L'analyse est déjà en cours. Veuillez patienter.")
            return
        
        self.deduplication_worker = DeduplicationWorker(df_active, self.settings)
        self.deduplication_worker.finished.connect(self.analysis_finished)
        self.deduplication_worker.progress.connect(self.update_progress_bar) 

        self.status_label.setText(f"Statut: ⏳ Analyse en cours sur catalogue '{self.current_catalog_key}'...")
        self.run_button.setEnabled(False)
        self.settings_button.setEnabled(False) 
        self.download_button.setEnabled(False)
        self.catalog_selector.setEnabled(False)
        self.comparison_text.setText("")
        self.deduplication_worker.start()

    def analysis_finished(self, success: bool, message: str):
        self.run_button.setEnabled(True)
        self.settings_button.setEnabled(True) 
        self.download_button.setEnabled(True)
        self.catalog_selector.setEnabled(True)
        self.update_progress_bar(0, "") 
        
        if success:
            self.load_report_into_table(OUTPUT_FILENAME)
            
            df_active = self.get_active_dataframe()
            if not self.report_df.empty and not df_active.empty:
                lignes_doublons_uniques = pd.concat([
                    self.report_df['LIGNE_1_NUMERO'].astype(str), 
                    self.report_df['LIGNE_2_NUMERO'].astype(str)
                ]).nunique()
                total_lines = len(df_active)
                
                info_msg = (
                    f"Analyse terminée avec succès. {len(self.report_df)} paires de doublons uniques trouvées.<br><br>"
                    f"**Lignes uniques du catalogue '{self.current_catalog_key}' impliquées : {lignes_doublons_uniques} / {total_lines}**"
                )
            else:
                info_msg = message

            QMessageBox.information(self, "Analyse terminée", info_msg)
        else:
            self.status_label.setText(f"Statut: ❌ Échec - {message}")
            QMessageBox.critical(self, "Erreur d'analyse", message)

    def generate_ground_truth(self):
        """ Génère un CSV avec les données complètes et les numéros de lignes liés. """
        active_df = self.get_active_dataframe()
        if active_df.empty or self.report_df.empty:
            QMessageBox.warning(self, "Erreur", "Veuillez d'abord charger un catalogue et lancer une analyse.")
            return

        # 1. Récupérer les index des lignes cochées dans le tableau
        checked_indices = []
        for row in range(self.report_model.rowCount()):
            item = self.report_model.item(row, 0)
            if item and item.checkState() == Qt.Checked:
                checked_indices.append(row)

        if not checked_indices:
            QMessageBox.warning(self, "Sélection", "Veuillez cocher au moins une ligne dans le tableau.")
            return

        # 2. Construire les données d'export
        gt_rows = []
        for idx in checked_indices:
            # On récupère les numéros de ligne identifiés par le rapport
            report_row = self.report_df.iloc[idx]
            line1_idx = int(report_row['LIGNE_1_NUMERO'])
            line2_idx = int(report_row['LIGNE_2_NUMERO'])

            # On crée deux entrées pour que chaque doublon soit référencé par l'autre
            for current, partner in [(line1_idx, line2_idx), (line2_idx, line1_idx)]:
                # Extraction de la ligne complète depuis le catalogue original
                full_data = active_df.iloc[current].to_dict()
                
                # On prépare la nouvelle ligne avec les colonnes de suivi en premier
                new_entry = {
                    "NUMERO_LIGNE_DOUBLON_ASSOCIEE": partner,
                    "MON_NUMERO_DE_LIGNE": current
                }
                new_entry.update(full_data) # On ajoute toutes les autres colonnes
                gt_rows.append(new_entry)

        # 3. Export en CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ground_truth_deduplication_{timestamp}.csv"
        
        try:
            pd.DataFrame(gt_rows).to_csv(
                filename, 
                index=False, 
                sep=OUTPUT_DELIMITER, 
                encoding='utf-8', 
                quotechar='"',
                quoting=csv.QUOTE_ALL
            )
            QMessageBox.information(self, "Succès", f"Fichier généré : {filename}\n({len(gt_rows)} lignes exportées)")
            self.status_label.setText(f"Statut: ✅ Ground Truth exporté ({filename})")
        except Exception as e:
            QMessageBox.critical(self, "Erreur d'export", f"Impossible d'écrire le fichier : {e}")

    def unpublish_button_clicked(self, line_index: int):
        """ Slot appelé lorsque l'un des boutons de dépublication est cliqué. """
        if line_index == 1:
            datasetid = self.current_datasetid_1
            line_num = self.current_line_num_1
        elif line_index == 2:
            datasetid = self.current_datasetid_2
            line_num = self.current_line_num_2
        else:
            return

        self.unpublish_dataset(datasetid, line_num)

    def unpublish_dataset(self, datasetid: Optional[str], line_num: Optional[int]):
        """ Lance l'opération de dépublication via le worker API. """
        if not datasetid or datasetid == 'N/A' or line_num is None:
            QMessageBox.warning(self, "Erreur UID", f"Ligne {line_num}: L'identifiant de jeu de données (datasetid) est manquant. (Vérifiez la présence de la colonne '{UID_COLUMN_NAME}' dans le catalogue)")
            return

        if self.api_action_worker and self.api_action_worker.isRunning():
            QMessageBox.warning(self, "Opération en cours", "Une opération de dépublication est déjà en cours. Veuillez patienter.")
            return

        # Confirmation par l'utilisateur
        reply = QMessageBox.question(self, 'Confirmation de Dépublication',
            f"Êtes-vous sûr de vouloir **DÉPUBLIER** la ligne {line_num} (datasetid: {datasetid}) ?\nCeci est une action potentiellement IRREVERSIBLE.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.status_label.setText(f"Statut: ⏳ Dépublication en cours pour la ligne {line_num}...")
            
            self.api_action_worker = ApiActionWorker(datasetid, line_num, API_KEY)
            self.api_action_worker.finished.connect(self.handle_unpublish_result)

            self.run_button.setEnabled(False)
            self.download_button.setEnabled(False)
            self.api_action_worker.start()

    def handle_unpublish_result(self, success: bool, message: str, line_num: int):
        """ Gère le résultat de l'opération de dépublication. """
        
        self.run_button.setEnabled(True)
        self.download_button.setEnabled(True)
        self.status_label.setText(f"Statut: {message}")

        if success:
            QMessageBox.information(self, "Dépublication Réussie", message)
        else:
            QMessageBox.critical(self, "Erreur de Dépublication", message)


    def show_comparison_view(self, index: QModelIndex, previous_index: QModelIndex = None):
        """ Affiche la vue de comparaison pour la ligne sélectionnée/focus. """
        
        df_active = self.get_active_dataframe()

        if not index.isValid():
            self.comparison_text.setText("Sélectionnez une ligne dans le rapport pour afficher la comparaison.")
            return

        if self.report_df.empty or df_active.empty:
            self.comparison_text.setText("Impossible de comparer : Le rapport ou le catalogue actif n'est pas chargé.")
            return

        report_row_index = index.row()
        selected_report_row = self.report_df.iloc[report_row_index]
        
        line1_num = int(selected_report_row['LIGNE_1_NUMERO'])
        line2_num = int(selected_report_row['LIGNE_2_NUMERO'])
        
        try:
            data1 = df_active.loc[line1_num]
            data2 = df_active.loc[line2_num]

            # --- Récupération et stockage des Dataset IDs et Line Numbers ---
            # Utilisation de la constante UID_COLUMN_NAME='datasetid'
            line1_datasetid = str(data1.get(UID_COLUMN_NAME, 'N/A'))
            line2_datasetid = str(data2.get(UID_COLUMN_NAME, 'N/A'))
            
            self.current_datasetid_1 = line1_datasetid
            self.current_datasetid_2 = line2_datasetid
            self.current_line_num_1 = line1_num
            self.current_line_num_2 = line2_num
            
            self.unpublish_btn_1.setText(f"Dépublier LIGNE 1 ({line1_num})")
            self.unpublish_btn_2.setText(f"Dépublier LIGNE 2 ({line2_num})")
            
            # Activation/Désactivation des boutons
            self.unpublish_btn_1.setEnabled(line1_datasetid != 'N/A')
            self.unpublish_btn_2.setEnabled(line2_datasetid != 'N/A')

            comparison_output = []
            
            self.comparison_title_label.setText(f"Vue de Comparaison (Textes Nettoyés): Paire {report_row_index + 1} (Catalogue: {self.current_catalog_key})")
            
            comparison_output.append(f"<h3 style='color: #007bff;'>Paire de Doublons: Ligne {line1_num} vs Ligne {line2_num}</h3>")
            comparison_output.append(f"<p><b>Niveau de Doublon:</b> <span style='color: {'red' if selected_report_row['NIVEAU_DOUBLON'] == 'Identité Totale' else 'orange'};'>{selected_report_row['NIVEAU_DOUBLON']}</span> | <b>Critère:</b> {selected_report_row['CRITERE_DETECTION']} | <b>Score:</b> {selected_report_row['SCORE_SIMILARITE']}</p>")
            comparison_output.append(f"<p><b>Dataset ID 1 (pour API):</b> <span style='color: {'green' if line1_datasetid != 'N/A' else 'red'};'>{line1_datasetid}</span></p>")
            comparison_output.append(f"<p><b>Dataset ID 2 (pour API):</b> <span style='color: {'green' if line2_datasetid != 'N/A' else 'red'};'>{line2_datasetid}</span></p>")
            comparison_output.append("<hr>")

            comparison_output.append("<table style='width:100%; border-collapse: collapse;'>")
            comparison_output.append("<thead><tr style='background-color: #f0f0f0;'>")
            
            # Largeur relative : 25% pour Champ, 37.5% pour Ligne 1 et Ligne 2
            comparison_output.append("<th style='width: 25%; text-align: left; padding: 5px; border: 1px solid #ccc;'>Champ</th>")
            comparison_output.append(f"<th style='width: 37.5%; text-align: left; padding: 5px; border: 1px solid #ccc; color: #28a745;'>LIGNE 1 ({line1_num})</th>")
            comparison_output.append(f"<th style='width: 37.5%; text-align: left; padding: 5px; border: 1px solid #ccc; color: #dc3545;'>LIGNE 2 ({line2_num})</th>") 
            comparison_output.append("</tr></thead><tbody>")
            
            comparison_fields = [
                ('default.title', "Titre", True),        
                ('default.description', "Description", True), 
                ('default.publisher', "Éditeur", True),  
                ('default.keyword', "Mots-clés", True), 
                ('default.references', "Référence", False), 
            ]
            
            # Styles des cellules <td> (visuels seulement)
            style_content_td = "border: 1px solid #eee; padding: 5px;"
            style_col_1 = f"width: 37.5%; color: #28a745; {style_content_td}"
            style_col_2 = f"width: 37.5%; color: #dc3545; {style_content_td}"
            
            style_pre_content = "margin: 0; white-space: pre-wrap; overflow-wrap: break-word; word-break: break-all;"

            for field_name, display_name, use_clean in comparison_fields:
                
                if use_clean:
                    col_name = f'{field_name}_CLEAN' 
                else:
                    col_name = field_name
                    
                val1 = str(data1.get(col_name, 'N/A'))
                val2 = str(data2.get(col_name, 'N/A'))
                
                val2_highlighted = highlight_diff(val1, val2)

                comparison_output.append("<tr>")
                comparison_output.append(f"<td style='border: 1px solid #eee; padding: 5px; font-weight: bold;'>{display_name}</td>")
                comparison_output.append(f"<td style='{style_col_1}'><pre style='{style_pre_content}'>{val1}</pre></td>") 
                comparison_output.append(f"<td style='{style_col_2}'><pre style='{style_pre_content}'>{val2_highlighted}</pre></td>") 
                comparison_output.append("</tr>")

            comparison_output.append("</tbody></table>")
            
            self.comparison_text.setHtml("".join(comparison_output))

        except KeyError:
             self.comparison_text.setText(f"Erreur: Impossible de trouver les lignes {line1_num} ou {line2_num} dans le catalogue actif. Veuillez relancer l'analyse.")
        except Exception as e:
            self.comparison_text.setText(f"Erreur inattendue lors de la comparaison : {e}")


# ==============================================================================
# EXÉCUTION DE L'APPLICATION
# ==============================================================================

if __name__ == "__main__":
    # Ajout de QLayout à la liste des imports PySide6 au début du fichier
    if 'fuzzywuzzy' not in sys.modules or 'pandas' not in sys.modules or 'PySide6' not in sys.modules or 'requests' not in sys.modules:
        print("----------------------------------------------------------------------")
        print("Veuillez installer les librairies requises :")
        print("pip install pandas pyside6 fuzzywuzzy python-Levenshtein requests")
        print("----------------------------------------------------------------------")
    
    if 'sklearn.feature_extraction.text' not in sys.modules:
        print("----------------------------------------------------------------------")
        print("Veuillez installer la librairie scikit-learn pour la règle TFIDF :")
        print("pip install scikit-learn")
        print("----------------------------------------------------------------------")
        
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())