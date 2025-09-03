import gc
import hashlib
import logging
import os
import re
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
import psutil
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import scipy

from ollama import Client

from .prompt import SYSTEM_PROMPT

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_LLM_BASE = os.environ.get("LLM_BASE_URL", "http://ollama:11434")
DEFAULT_LLM_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

# Performance optimized parameters
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 2
SNIPPET_SIZE = 300
CANDIDATES = 30
EMB_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_CHUNKS_PER_TABLE = 3000
CACHE_SIZE = 100  # Cache recent queries

# Query result cache
_query_cache = {}


def _cache_query_result(query_hash: str, result: tuple) -> None:
    """Cache query results

    Args:
        query_hash (str): Unique identifier (hash) of the query to be cached
        result (tuple): The query result to cache
    """
    if len(_query_cache) >= CACHE_SIZE:
        # Remove oldest entry
        oldest_key = next(iter(_query_cache))
        del _query_cache[oldest_key]

    _query_cache[query_hash] = result


def _get_cached_result(query_hash: str) -> tuple | None:
    """Retrieve a cached query result if available

    Args:
        query_hash (str): Unique identifier (hash) of the query

    Returns:
        tuple | None: The cached query result if present, otherwise None
    """
    return _query_cache.get(query_hash)


def _get_query_hash(query: str) -> str:
    """Generate a stable hash for a query string

    Args:
        query (str): The query string to hash

    Returns:
        str: The MD5 hash of the normalized query string
    """
    return hashlib.md5(query.lower().strip().encode()).hexdigest()


def log_memory_usage(stage: str) -> None:
    """Log the current process and system memory usage

    Args:
        stage (str): A label describing the current stage of execution
    """

    try:
        # Force garbage collection before measuring
        gc.collect()

        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        logger.info(f"Memory usage at {stage}: {memory_mb:.2f} MB")

        # Log system memory
        sys_memory = psutil.virtual_memory()
        logger.info(
            f"System memory at {stage}: {sys_memory.percent:.1f}% used, {sys_memory.available / 1024 / 1024 / 1024:.2f} GB available"
        )

        # Force more aggressive cleanup if memory is high
        if memory_mb > 4000:
            logger.warning(
                f"High memory usage detected ({memory_mb:.2f} MB), forcing cleanup"
            )
            gc.collect()

    except Exception as e:
        logger.error(f"Error logging memory usage: {e}")


def cleanup_memory() -> None:
    """Force garbage collection and clear cache to free memory"""
    try:
        # Clear any cached data that might be hanging around
        if hasattr(st, "cache_data"):
            st.cache_data.clear()
        if hasattr(st, "cache_resource"):
            st.cache_resource.clear()

        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()

        logger.info("Memory cleanup completed")
    except Exception as e:
        logger.error(f"Error during memory cleanup: {e}")


def request_ollama(
    messages: list[dict[str, str]],
    base_url: str = DEFAULT_LLM_BASE,
    model: str = DEFAULT_LLM_MODEL,
    temperature: float = 0.2,
) -> str:
    """Handling request to ollama API

    Args:
        messages (list[dict[str, str]]): System and user message to instruct the llm and provide context.
        base_url (str, optional): Base URL for the llm API. Defaults to DEFAULT_LLM_BASE.
        model (str, optional): Model to use for the llm. Defaults to DEFAULT_LLM_MODEL.
        temperature (float, optional): Temperature for the llm response. Defaults to 0.2.

    Returns:
        str: The response from the llm API.
    """
    log_memory_usage("before_ollama_request")
    logger.info(f"Requesting Ollama with model: {model}, base_url: {base_url}")

    try:
        client = Client(host=base_url, headers={"x-some-header": "some-value"})
        response = client.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature},
            stream=False,
        )

        log_memory_usage("after_ollama_request")
        logger.info("Ollama request completed successfully")
        return response.message.content or ""
    except Exception as e:
        logger.error(f"Ollama request failed: {e}")
        log_memory_usage("after_ollama_error")
        raise


def _detect_columns(df: pd.DataFrame) -> dict[str, str | None]:
    """Heuristically detect relevant column names in a dataframe

    Args:
        df (pd.DataFrame): Input dataframe to inspect

    Returns:
        dict[str, str | None]: Mapping with keys {"text","date","speaker", "title","id"} to the original dataframe column names if found, otherwise None for missing fields
    """

    cols = {c.lower(): c for c in df.columns}

    def _pick(cands: Iterable[str]) -> str | None:
        for k in cands:
            if k in cols:
                return cols[k]

        for k in cols:
            if any(n in k for n in cands):
                return cols[k]

        return None

    return {
        "text": _pick(
            [
                "text",
                "content",
                "speech",
                "speech_text",
                "transcript",
                "rede",
                "rede_text",
                "beitrag_text",
            ]
        ),
        "date": _pick(
            ["date", "sitzungsdatum", "speech_date", "timestamp", "datetime"]
        ),
        "speaker": _pick(["speaker", "redner", "name", "speaker_name", "abgeordneter"]),
        "title": _pick(
            [
                "title",
                "topic",
                "subject",
                "tagesordnungspunkt",
                "agenda",
                "ueberschrift",
            ]
        ),
        "id": _pick(["id", "speech_id", "rede_id", "doc_id", "beitrag_id"]),
    }


def _chunk_text(
    t: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Chunk text into fixed-size segments with character overlap

    Args:
        t (str): The input text to split
        size (int, optional): Maximum characters per chunk. Defaults to CHUNK_SIZE.
        overlap (int, optional): Characters of overlap between consecutive chunks. Defaults to CHUNK_OVERLAP.

    Returns:
        list[str]: List of chunk strings
    """
    t = re.sub(r"\s+", " ", str(t)).strip()
    if not t:
        return []
    if len(t) <= size:
        return [t]

    chunks, start = [], 0
    while start < len(t):
        end = min(len(t), start + size)
        chunk = t[start:end]

        # Skip very short chunks at the end
        if len(chunk.strip()) < 50:
            break

        chunks.append(chunk)
        if end == len(t):
            break
        start = end - overlap

        # Limit number of chunks per text to prevent memory issues
        if len(chunks) >= 10:
            break

    return chunks


def _make_snippet(text: str, query: str, max_chars: int = SNIPPET_SIZE) -> str:
    """Create a short preview centered around the first query hit

    Args:
        text (str): Source text to summarize
        query (str): User query used to locate highlight terms
        max_chars (int, optional): Maximum length of the snippet. Defaults to SNIPPET_SIZE.

    Returns:
        str: The created snippet
    """

    if len(text) <= max_chars:
        return text
    terms = [w for w in re.findall(r"\w+", query, flags=re.I) if len(w) > 2]
    if not terms:
        return text[:max_chars] + "…"
    lower = text.lower()
    positions = [lower.find(t.lower()) for t in terms]
    positions = [p for p in positions if p >= 0]
    if not positions:
        return text[:max_chars] + "…"
    pos = min(positions)
    half = max_chars // 2
    start = max(0, pos - half)
    end = min(len(text), start + max_chars)
    return f"{'…' if start>0 else ''}{text[start:end]}{'…' if end<len(text) else ''}"


def _session_signature() -> tuple:
    """Produce a lightweight signature of current session tables

    Returns:
        tuple: A tuple of per-table signatures, or an empty tuple if no tables are registered in session state.
    """
    if "db_tables" not in st.session_state:
        return ()
    sig_parts = []
    for name in st.session_state["db_tables"]:
        df = st.session_state.get(name)
        if isinstance(df, pd.DataFrame) and not df.empty:
            sig_parts.append(
                (
                    name,
                    len(df),
                    tuple(df.columns),
                )
            )
        else:
            sig_parts.append((name, 0, ()))
    return tuple(sig_parts)


@st.cache_data(show_spinner=False)
def _build_chunked_corpus(session_sig: tuple) -> pd.DataFrame:
    """Construct a chunk-level corpus dataframe from session tables

    Args:
        session_sig (tuple): Output of `_session_signature()` describing the tables to process

    Returns:
        pd.DataFrame: A dataframe with columns: ["doc_id","date","speaker","title","source_table","chunk"].
            Empty if no valid chunks were produced.
    """
    log_memory_usage("start_build_corpus")
    logger.info(f"Building chunked corpus for {len(session_sig)} tables")

    rows: list[dict[str, Any]] = []
    total_chunks = 0

    if not session_sig:
        return pd.DataFrame()

    for name, _, _ in session_sig:
        df = st.session_state.get(name)
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue

        logger.info(f"Processing table: {name} with {len(df)} rows")

        # Limit rows per table to prevent memory overflow
        if len(df) > 50000:  # If table is very large, sample it
            logger.warning(f"Table {name} has {len(df)} rows, sampling 50000 rows")
            df = df.sample(n=50000, random_state=42).reset_index(drop=True)
            gc.collect()  # Clean up the original large dataframe

        cols = _detect_columns(df)
        text_col = cols["text"]
        if not text_col or text_col not in df.columns:
            logger.warning(f"No text column found for table: {name}")
            continue

        keep = [
            c
            for c in [
                cols["id"],
                text_col,
                cols["date"],
                cols["speaker"],
                cols["title"],
            ]
            if c
        ]

        # Process in batches to reduce memory usage
        batch_size = 1000
        table_chunks = 0

        for start_idx in range(0, len(df), batch_size):
            if total_chunks >= MAX_CHUNKS_PER_TABLE:
                logger.warning(
                    f"Reached maximum chunk limit, stopping at {total_chunks} chunks"
                )
                break

            end_idx = min(start_idx + batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]

            slim = batch_df[keep].rename(
                columns={
                    (cols["id"] or "id"): "doc_id",
                    text_col: "text",
                    (cols["date"] or "date"): "date",
                    (cols["speaker"] or "speaker"): "speaker",
                    (cols["title"] or "title"): "title",
                }
            )

            # normalize date to YYYY-MM-DD strings
            if "date" in slim.columns:
                with pd.option_context("mode.chained_assignment", None):
                    try:
                        slim["date"] = pd.to_datetime(
                            slim["date"], errors="coerce"
                        ).dt.date.astype(str)
                    except Exception:
                        slim["date"] = slim["date"].astype(str)

            slim["source_table"] = name

            for _, r in slim.iterrows():
                if total_chunks >= MAX_CHUNKS_PER_TABLE:
                    break

                text = str(r.get("text") or "").strip()
                if not text:
                    continue

                # Limit text length to prevent memory issues
                if len(text) > 10000:  # Truncate very long texts
                    text = text[:10000] + "..."

                for ch in _chunk_text(text):
                    if total_chunks >= MAX_CHUNKS_PER_TABLE:
                        break

                    rows.append(
                        {
                            "doc_id": str(r.get("doc_id") or ""),
                            "date": r.get("date") or "",
                            "speaker": r.get("speaker") or "",
                            "title": r.get("title") or "",
                            "source_table": name,
                            "chunk": ch,
                        }
                    )
                    total_chunks += 1
                    table_chunks += 1

            # Clean up batch
            del batch_df, slim

            # Force garbage collection every 10 batches
            if (start_idx // batch_size) % 10 == 0:
                gc.collect()
                log_memory_usage(f"build_corpus_batch_{start_idx}")

        logger.info(f"Table {name}: processed {table_chunks} chunks")

        if total_chunks >= MAX_CHUNKS_PER_TABLE:
            logger.warning(f"Reached maximum chunk limit for session")
            break

    result_df = pd.DataFrame(rows)
    logger.info(f"Built corpus with {len(result_df)} chunks")
    log_memory_usage("end_build_corpus")

    # Final cleanup
    gc.collect()
    return result_df


@st.cache_resource(show_spinner=False, max_entries=3)
def _build_tfidf_index(
    chunk_df: pd.DataFrame,
) -> tuple[TfidfVectorizer, scipy.sparse.csr_matrix] | tuple[None, None]:
    """Build a TF-IDF vectorizer and matrix for the chunk corpus.

    Args:
        chunk_df (pd.DataFrame): Corpus produced by `_build_chunked_corpus` containing a "chunk" column.

    Returns:
        tuple[TfidfVectorizer, scipy.sparse.csr_matrix] | tuple[None, None]: The fitted vectorizer and the TF-IDF matrix. Returns (None, None) if `chunk_df` is empty.
    """
    if chunk_df is None or chunk_df.empty:
        return None, None

    log_memory_usage("start_tfidf_build")
    logger.info(f"Building TF-IDF index for {len(chunk_df)} chunks")

    # Optimized TF-IDF parameters for better performance
    vect = TfidfVectorizer(
        ngram_range=(1, 2),
        max_df=0.95,
        min_df=1,
        max_features=5000,
        strip_accents="unicode",
        lowercase=True,
        stop_words=None,
    )

    chunks = chunk_df["chunk"].tolist()

    # Build index in smaller batches if dataset is large
    if len(chunks) > 5000:
        logger.info("Large dataset detected, building TF-IDF index in optimized mode")

        # Sample for vocabulary building to speed up the process
        sample_chunks = chunks[:: max(1, len(chunks) // 2000)]
        vect.fit(sample_chunks)
        tfidf = vect.transform(chunks)
    else:
        tfidf = vect.fit_transform(chunks)

    log_memory_usage("end_tfidf_build")
    logger.info(f"TF-IDF matrix shape: {tfidf.shape}")
    return vect, tfidf


@st.cache_resource(show_spinner=False)
def _get_emb_model() -> None | SentenceTransformer:
    """Load and cache the embedding model.

    Returns:
        None | SentenceTransformer: The loaded model, or None if loading fails.
    """
    log_memory_usage("start_load_embedding_model")
    logger.info(f"Loading embedding model: {EMB_MODEL_NAME}")
    model = SentenceTransformer(EMB_MODEL_NAME)
    log_memory_usage("end_load_embedding_model")
    return model


def retrieve_context(query: str) -> tuple[str, list[dict[str, Any]]]:
    """Retrieve context passages relevant to a query

    Args:
        query (str): The user query to search for.

    Returns:
        tuple[str, list[dict[str, Any]]]:
            - A concatenated context string (possibly empty).
            - A list of source dicts with keys:
                {"rank","date","preview","similarity","doc_id","speaker","title","source_table"}.
    """
    log_memory_usage("start_retrieve_context")
    logger.info(f"Retrieving context for query: {query[:100]}...")

    # Check cache first
    query_hash = _get_query_hash(query)
    cached_result = _get_cached_result(query_hash)
    if cached_result is not None:
        logger.info("Retrieved result from cache")
        return cached_result

    q = (query or "").strip()
    if not q:
        return "", []

    # build/refresh corpus & indices from session_state
    session_sig = _session_signature()
    chunk_df = _build_chunked_corpus(session_sig)
    if chunk_df is None or chunk_df.empty:
        logger.warning("No chunks available for retrieval")
        return "", []

    # Limit chunk_df size for performance
    if len(chunk_df) > 10000:
        logger.info(
            f"Limiting chunk_df from {len(chunk_df)} to 10000 rows for performance"
        )
        chunk_df = chunk_df.head(10000)

    vect, tfidf = _build_tfidf_index(chunk_df)
    if vect is None or tfidf is None:
        logger.warning("TF-IDF index could not be built")
        return "", []

    # TF-IDF recall
    log_memory_usage("start_tfidf_search")
    q_vec = vect.transform([q])
    sims = linear_kernel(q_vec, tfidf).ravel()
    if sims.max(initial=0) <= 0:
        logger.info("No TF-IDF matches found")
        return "", []

    # Get top candidates more efficiently
    top_indices = np.argpartition(sims, -CANDIDATES)[-CANDIDATES:]
    top_indices = top_indices[sims[top_indices] > 0]

    if len(top_indices) == 0:
        logger.info("No candidate indices found")
        return "", []

    # Sort by similarity score
    top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

    logger.info(f"Found {len(top_indices)} TF-IDF candidates")
    log_memory_usage("end_tfidf_search")

    # Semantic rerank
    log_memory_usage("start_semantic_rerank")
    emb_model = _get_emb_model()
    if emb_model is not None:
        logger.info("Performing semantic reranking")
        cand_texts = chunk_df.iloc[top_indices]["chunk"].tolist()

        # Process in smaller batches to reduce memory usage
        batch_size = min(16, len(cand_texts))
        q_emb = emb_model.encode([q], normalize_embeddings=True)
        c_emb = emb_model.encode(
            cand_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        sem_scores = (c_emb @ q_emb.T).ravel()

        # Clean up intermediate variables
        del cand_texts, q_emb, c_emb
        gc.collect()

        order = np.argsort(sem_scores)[::-1]
        chosen = [
            (top_indices[i], float(sem_scores[i]), float(sims[top_indices[i]]))
            for i in order[:TOP_K]
        ]
    else:
        logger.info("Using TF-IDF scores only (no semantic model)")
        chosen = [(i, float(sims[i]), float(sims[i])) for i in top_indices[:TOP_K]]

    log_memory_usage("end_semantic_rerank")

    if not chosen:
        logger.warning("No final candidates chosen")
        return "", []

    # Build context + sources
    context_blocks: list[str] = []
    sources: list[dict[str, Any]] = []

    for rank, (i, sem_score, tfidf_score) in enumerate(chosen, start=1):
        row = chunk_df.iloc[int(i)]
        date_br = f"[{row['date']}]" if row.get("date") else ""
        header_bits = [b for b in [row.get("speaker", ""), row.get("title", "")] if b]
        header = " — ".join(header_bits)
        snippet = _make_snippet(row["chunk"], q, SNIPPET_SIZE)
        block = f"{header} {date_br}\n{snippet}" if header else f"{date_br}\n{snippet}"
        context_blocks.append(block)

        sources.append(
            {
                "rank": rank,
                "date": row.get("date", ""),
                "preview": snippet,
                "similarity": round(
                    float(sem_score if emb_model is not None else tfidf_score), 4
                ),
                "doc_id": row.get("doc_id", ""),
                "speaker": row.get("speaker", ""),
                "title": row.get("title", ""),
                "source_table": row.get("source_table", ""),
            }
        )

    context = "\n\n".join(context_blocks)
    result = (context, sources) if context.strip() else ("", [])

    # Cache result
    try:
        _cache_query_result(query_hash, result)
        logger.info("Cached query result for future use")
    except Exception as e:
        logger.warning(f"Failed to cache result: {e}")

    log_memory_usage("end_retrieve_context")
    logger.info(
        f"Retrieved {len(sources)} sources with {len(context)} characters of context"
    )
    return result


def answer_with_retrieval(
    user_query: str, system_prompt: str = SYSTEM_PROMPT
) -> tuple[str, list[dict[str, Any]]]:
    """Generate an answer using retrieval-augmented generation

    Args:
        user_query (str): The user’s question
        system_prompt (str, optional): System instruction prefix passed to the LLM. Defaults to SYSTEM_PROMPT.

    Returns:
        tuple[str, list[dict[str, Any]]]: The model’s answer text and the list of source metadata returned by `retrieve_context`
    """
    log_memory_usage("start_answer_with_retrieval")
    logger.info(f"Starting answer generation for query: {user_query[:100]}...")

    context, sources = retrieve_context(user_query)

    found_context = bool(context and context.strip())
    if found_context:
        user_content = f"Context:\n{context}\n\nQuestion: {user_query}"
        logger.info(
            f"Using context of {len(context)} characters with {len(sources)} sources"
        )
    else:
        # No context -> allow a general answer and be explicit about limitations
        system_prompt = (
            system_prompt
            + "say that you couldn't find relevant context and offer to refine the query."
        )
        user_content = user_query
        logger.info("No context found, generating general response")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    try:
        answer = request_ollama(messages)
        logger.info("Answer generation completed successfully")

    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        answer = f"LLM-Fehler: {e}"

    log_memory_usage("end_answer_with_retrieval")
    return answer, sources
