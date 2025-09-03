# this file contains the approach of utilizing BERTopic for topic modeling
import os

import hdbscan
import nltk
import pandas as pd
import umap
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

nltk.download("stopwords")
nltk.download("wordnet")

DATA_DIR = os.path.join("app", "data")

speeches = pd.read_parquet(os.path.join(DATA_DIR, "speeches.parquet"))
contributions = pd.read_parquet(os.path.join(DATA_DIR, "contributions.parquet"))
politicians = pd.read_parquet(os.path.join(DATA_DIR, "politicians.parquet"))
factions = pd.read_parquet(os.path.join(DATA_DIR, "factions.parquet"))
electoral_terms = pd.read_parquet(os.path.join(DATA_DIR, "electoral_terms.parquet"))

politicians.to_parquet(os.path.join(DATA_DIR, "politicians.parquet"))


speeches = speeches.loc[speeches["electoral_term"] == 20]

speeches["session"].nunique()

## Topic Modelling approach

stop_words = set(stopwords.words("german"))
domain_stopwords = {
    "herr",
    "frau",
    "präsident",
    "präsidentin",
    "vizepräsident",
    "werte",
    "verehrte",
    "liebe",
    "geehrte",
    "kollege",
    "kollegen",
    "kollegin",
    "kolleginnen",
    "damen",
    "herren",
    "bundestag",
    "bundesregierung",
    "bundeskanzler",
    "bundeskanzlerin",
    "minister",
    "ministerin",
    "abgeordnete",
    "abgeordneter",
    "abgeordneten",
    "fraktion",
    "parlament",
    "ausschuss",
    "drucksache",
    "bitte",
    "danke",
    "ja",
    "nein",
    "also",
    "nun",
    "heute",
    "morgen",
    "gestern",
    "gerade",
    "erstens",
    "zweitens",
    "drittens",
    "überhaupt",
    "wirklich",
    "vielleicht",
    "natürlich",
    "meine",
    "sehr",
    "vielen",
    "herzlich",
    "willkommen",
    "deutlich",
    "wichtig",
    "notwendig",
    "sicherlich",
    "offensichtlich",
    "selbstverständlich",
    "ohnehin",
    "müssen",
    "müßte",
    "müsste",
    "könnte",
    "dürfte",
    "sollte",
    "dafür",
    "brauchen",
}

stop_words = list(stop_words.union(domain_stopwords))


_docs_df = speeches[["id", "speech_content"]].dropna().copy(deep=True)
_docs_df["text"] = _docs_df["speech_content"].astype(str)

# remove empty speeches
_docs_df = _docs_df.loc[_docs_df["text"].str.strip() != ""]

_docs_df["word_count"] = _docs_df["text"].str.split().str.len()

_docs_df = _docs_df.loc[_docs_df["word_count"] > 50]

_docs_df["has_contribution"] = _docs_df["id"].isin(contributions["speech_id"])

# remove {{}} braces from text if present
_docs_df["text_cleaned"] = _docs_df["text"].str.replace(r"\({.*?}\)", " ", regex=True)

# add lemmatization
lemmatizer = nltk.stem.WordNetLemmatizer()
_docs_df["text_lemmatized"] = _docs_df["text_cleaned"].apply(
    lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()])
)

vectorizer_model = CountVectorizer(
    stop_words=stop_words,
    lowercase=True,
    min_df=5,
    ngram_range=(1, 3),
)

embedding_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
embedding_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

rep_model = KeyBERTInspired(nr_candidate_words=30)

topic_model = BERTopic(
    language="german",
    calculate_probabilities=True,
    vectorizer_model=vectorizer_model,
    embedding_model=embedding_model,
    representation_model=rep_model,
    verbose=True,
)

topics, probs = topic_model.fit_transform(_docs_df["text_lemmatized"].tolist())

topic_info = topic_model.get_topic_info()


# Using predefined seed topics
# ai generated list
seed_topic_list = [
    # Haushalt/Finanzen
    [
        "haushalt",
        "etat",
        "haushaltsgesetz",
        "mittel",
        "bundeshaushalt",
        "ausgaben",
        "einnahmen",
    ],
    # Migration/Asyl
    [
        "migration",
        "asyl",
        "flüchtling",
        "einwanderung",
        "aufenthalt",
        "integration",
        "schutzsuchende",
    ],
    # Klima/Energie
    [
        "klima",
        "klimaschutz",
        "co2",
        "energie",
        "erneuerbare",
        "windkraft",
        "photovoltaik",
        "energiewende",
    ],
    # Wirtschaft/Standort
    [
        "wirtschaft",
        "unternehmen",
        "mittelstand",
        "standort",
        "industriepolitik",
        "konjunktur",
    ],
    # Außen/Sicherheit
    [
        "außenpolitik",
        "verteidigung",
        "bundeswehr",
        "nato",
        "russland",
        "ukraine",
        "sicherheitspolitik",
    ],
    # Arbeit/Soziales
    [
        "arbeit",
        "arbeitsmarkt",
        "tarif",
        "rente",
        "sozial",
        "grundsicherung",
        "bafög",
        "familienpolitik",
    ],
    # Gesundheit/Pflege
    [
        "gesundheit",
        "pflege",
        "krankenhaus",
        "gesetzliche_krankenversicherung",
        "pflegeversicherung",
    ],
    # Digital/Daten
    [
        "digitalisierung",
        "daten",
        "ki",
        "künstliche_intelligenz",
        "cybersicherheit",
        "breitband",
        "digitalstrategie",
    ],
    # Bildung/Forschung
    [
        "bildung",
        "schule",
        "hochschule",
        "forschung",
        "wissenschaft",
        "exzellenzstrategie",
    ],
    # Verkehr/Mobilität
    ["verkehr", "mobilität", "bahn", "deutsche_bahn", "autoindustrie", "ladestation"],
]

vectorizer_model = CountVectorizer(
    stop_words=list(stop_words),
    lowercase=True,
    ngram_range=(1, 3),
    max_df=0.95,
    min_df=5,
)

embedding_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

# reduce dimensionality and improve clustering
umap_model = umap.UMAP(
    n_neighbors=40,
    n_components=5,
    min_dist=0.1,
    metric="cosine",
    random_state=42,
)
hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=120,
    min_samples=30,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True,
)

rep_model = KeyBERTInspired(nr_candidate_words=30)

topic_model = BERTopic(
    language="german",
    embedding_model=embedding_model,
    vectorizer_model=vectorizer_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    representation_model=rep_model,
    seed_topic_list=seed_topic_list,
    calculate_probabilities=True,
    verbose=True,
)

docs_for_fit = _docs_df["text_cleaned"].tolist()
topics, probs = topic_model.fit_transform(docs_for_fit)

topic_model.get_topic_info()
