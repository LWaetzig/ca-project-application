import re
import spacy
import pandas as pd
import os
from tqdm import tqdm

DATA_DIR = os.path.join("app", "data")

speeches = pd.read_parquet(os.path.join(DATA_DIR, "speeches.parquet"))
speeches = speeches.loc[speeches["electoral_term"] == 20]

contributions = pd.read_parquet(os.path.join(DATA_DIR, "contributions.parquet"))
contributions = contributions.loc[contributions["speech_id"].isin(speeches["id"])][
    ["id", "type", "content", "speech_id", "text_position"]
]
politicians = pd.read_csv(os.path.join(DATA_DIR, "members.csv"))

speeches_with_contrib = speeches.loc[speeches["id"].isin(contributions["speech_id"])]
speeches_with_no_contrib = speeches.loc[
    ~speeches["id"].isin(contributions["speech_id"])
]

INTERVENTION_PATTERN = r"\(\{.*?\}\)"

nlp = spacy.load("de_core_news_sm")


def _split_by_interventions_then_sentences(
    text: str, pattern: str = INTERVENTION_PATTERN
) -> tuple[list[str], list[int]]:
    """
    Zuerst Text an Interventions-Markern (z. B. "({...})") aufteilen, danach
    nur die Nicht-Interventions-Teile in Sätze segmentieren. Gibt zurück:
        - sentences: flache Liste aller Sätze ohne Interventionsmarker
        - intervention_boundaries: Liste von Satzindizes, an denen eine Intervention zwischen Sätzen lag
        (d. h. Position ist die Anzahl Sätze, die vor der Intervention bereits gesammelt wurden).
    """
    # Splitten und Delimiter behalten
    chunks = re.split(f"({pattern})", text)
    sentences: list[str] = []
    intervention_boundaries: list[int] = []

    for chunk in chunks:
        if not chunk or chunk.strip() == "":
            continue
        # Wenn der Chunk selbst ein Interventionsmarker ist -> Boundary merken
        if re.fullmatch(pattern, chunk):
            intervention_boundaries.append(len(sentences))
            continue
        # Sonst: normalen Text in Sätze zerlegen und anhängen
        doc = nlp(chunk)
        for sent in doc.sents:
            s = sent.text.strip()
            if s:
                sentences.append(s)

    return sentences, intervention_boundaries


def _build_windows_from_boundaries(
    sentences: list[str], boundaries: list[int], window: int = 2
) -> list[dict]:
    """
    Erzeugt für jede Intervention (Boundary zwischen Sätzen) ein Pre-/Post-Fenster.
    Für eine Boundary b gilt:
        - Pre = Sätze[max(0, b-window): b]
        - Post = Sätze[b: b+window]
    """
    results: list[dict] = []
    for b in boundaries:
        pre = sentences[max(0, b - window) : b]
        post = sentences[b : b + window]
        results.append(
            {
                "boundary": b,
                "pre": pre,
                "post": post,
            }
        )
    return results


from transformers import pipeline

_sentiment = pipeline("sentiment-analysis", model="oliverguhr/german-sentiment-bert")


def score_sentences(sents: list[str]) -> list[float]:
    if not sents:
        return []
    out = _sentiment(sents)
    scores: list[float] = []
    for r in out:
        label = r.get("label", "neutral").lower()
        score = float(r.get("score", 0.0))
        if label == "positive":
            scores.append(score)
        elif label == "negative":
            scores.append(-score)
        else:
            scores.append(0.0)
    return scores


cords = list()
WINDOW = 2
counter = 0

for _, row in tqdm(speeches_with_contrib.iterrows(), total=len(speeches_with_contrib)):
    sentences, boundaries = _split_by_interventions_then_sentences(
        row["speech_content"]
    )
    windows = _build_windows_from_boundaries(sentences, boundaries, window=WINDOW)
    party = politicians.loc[
        politicians["id"] == row["politician_id"], "partei_kurz"
    ].values[0]

    for w in windows:
        pre_scores = score_sentences(w["pre"])
        post_scores = score_sentences(w["post"])
        if pre_scores and post_scores:
            pre_mean = sum(pre_scores) / len(pre_scores)
            post_mean = sum(post_scores) / len(post_scores)
            cords.append(
                {
                    "speech_id": row["id"],
                    "politician_id": row["politician_id"],
                    "politician": f"{row['first_name']} {row['last_name']}",
                    "party": party,
                    "intervention_type": "",
                    "pre_mean": pre_mean,
                    "post_mean": post_mean,
                    "delta": post_mean - pre_mean,
                }
            )

    counter += 1
    if counter == 50:
        break

results = pd.DataFrame(cords)
