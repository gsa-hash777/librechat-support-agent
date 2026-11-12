from collections import Counter
import math
import re


def tokenize(text: str) -> list[str]:
    """Chinese overlapping bigrams plus whole ASCII words; no model download."""
    tokens = []
    for part in re.findall(r"[a-z0-9_]+|[\u3400-\u9fff]+", text.lower()):
        if re.fullmatch(r"[\u3400-\u9fff]+", part):
            tokens.extend(part[index:index + 2] for index in range(max(1, len(part) - 1)))
        else:
            tokens.append(part)
    return tokens


def search_documents(documents: list[dict], query: str, limit: int = 5) -> list[dict]:
    query_tokens = set(tokenize(query))
    if not documents or not query_tokens:
        return []
    # A single BM25 index over weighted title + content, recomputed for this small demo corpus.
    indexed = [(document, Counter(tokenize(document["title"] + " " + document["title"] + " " + document["content"]))) for document in documents]
    average_length = sum(sum(tokens.values()) for _, tokens in indexed) / len(indexed)
    frequencies = Counter(token for _, tokens in indexed for token in tokens)
    results = []
    k1, b = 1.5, 0.75
    for document, tokens in indexed:
        score = 0.0
        length = sum(tokens.values())
        for token in query_tokens & tokens.keys():
            frequency = tokens[token]
            inverse_frequency = math.log(1 + (len(indexed) - frequencies[token] + 0.5) / (frequencies[token] + 0.5))
            score += inverse_frequency * frequency * (k1 + 1) / (frequency + k1 * (1 - b + b * length / average_length))
        if score > 0:
            results.append({"id": document["id"], "title": document["title"], "content": document["content"], "source": document["source"], "score": round(score, 6)})
    return sorted(results, key=lambda item: (-item["score"], item["id"]))[:limit]
