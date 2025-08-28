import os
import pandas as pd
from app.src.Database import Database
from dotenv import load_dotenv

load_dotenv("app/.env")

database = Database(
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    database=os.getenv("DB_NAME"),
)


# get all tables content in the database
speeches = database.fetch_data("SELECT * FROM open_discourse.speeches;")
contributions = database.fetch_data(
    "SELECT * FROM open_discourse.contributions_extended;"
)
politicians = database.fetch_data("SELECT * FROM open_discourse.politicians;")
electoral_terms = database.fetch_data("SELECT * FROM open_discourse.electoral_terms;")
factions = database.fetch_data("SELECT * FROM open_discourse.factions;")

electoral_terms["start_date"] = pd.to_datetime(electoral_terms["start_date"])
electoral_terms["end_date"] = pd.to_datetime(electoral_terms["end_date"])

speeches.to_parquet(os.path.join("app", "data", "speeches.parquet"))
contributions.to_parquet(os.path.join("app", "data", "contributions.parquet"))
politicians.to_parquet(os.path.join("app", "data", "politicians.parquet"))
electoral_terms.to_parquet(os.path.join("app", "data", "electoral_terms.parquet"))
factions.to_parquet(os.path.join("app", "data", "factions.parquet"))

speeches = database.fetch_data(
    "SELECT * FROM open_discourse.speeches WHERE electoral_term = 20;"
).sort_values(by=["id"])

politicians = database.fetch_data(
    "SELECT * FROM open_discourse.politicians;"
).sort_values(by=["id"])


query = "SELECT * FROM open_discourse.contributions_extended;"  # get all contributions
contributions = database.fetch_data(query)

contributions["type"].value_counts()

# use speeches to count how many of them include contributions
speeches_with_contributions = contributions["speech_id"].nunique()
speeches_total = speeches["id"].nunique()

speeches["has_contribution"] = speeches["id"].isin(contributions["speech_id"])
speeches["has_contribution"].value_counts()

speeches_with_contrib = speeches.loc[speeches["has_contribution"] == True][
    ["id", "first_name", "last_name", "politician_id", "speech_content"]
]


from transformers import pipeline
import re

sentiment = pipeline("sentiment-analysis", model="oliverguhr/german-sentiment-bert")
test_speech = speeches_with_contrib.iloc[0]["speech_content"]

# split text by contributions {{}}
test_speech = re.split(r"\(\{.*?\}\)", test_speech)

test = sentiment(test_speech[0])
sentiment(test_speech[1])
sentiment(test_speech[2])


import re
import spacy
from transformers import pipeline

nlp = spacy.load("de_core_news_sm")


def segment_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


segment_sentences(test_speech[0])

sentiment(segment_sentences(test_speech[0])[-1])
sentiment(segment_sentences(test_speech[1])[0])

print(test_speech)


## Topic Modelling

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
import umap
import hdbscan
from bertopic.representation import KeyBERTInspired

nltk.download("stopwords")
nltk.download("wordnet")

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

speeches = database.fetch_data(
    "SELECT * FROM open_discourse.speeches WHERE electoral_term = 20;"
)
contributions = database.fetch_data(
    "SELECT * FROM open_discourse.contributions_extended"
)

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


for doc in _docs_df.iterrows()[:1000]:
    print(doc)
    print(len(doc[1]["text_lemmatized"].split()))
    print("-------")


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


# Beispiel: Seed-Listen für Bundestagsthemen
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

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import umap
import hdbscan


vectorizer_model = CountVectorizer(
    stop_words=list(stop_words),  # deine erweiterte Stoppliste
    lowercase=True,
    ngram_range=(1, 3),  # Mehrwortbegriffe zulassen
    max_df=0.95,
    min_df=5,  # ggf. an Korpusgröße anpassen
)

# 2) Embeddings – für Deutsch oft eine Stufe stärker als MiniLM:
embedding_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

# 3) Dimensionalität/Clustering (stabilere, größere Themen)
umap_model = umap.UMAP(
    n_neighbors=40,
    n_components=5,
    min_dist=0.1,
    metric="cosine",
    random_state=42,
)
hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=120,  # größer -> weniger Mini-Cluster
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

# 6) Fit – nutze dokumente MIT Phrasen, falls vorhanden
docs_for_fit = _docs_df["text_cleaned"].tolist()
topics, probs = topic_model.fit_transform(docs_for_fit)

topic_model.get_topic_info()
