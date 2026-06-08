"""Production client helpers for OpenAI, Weaviate, Jina and PageIndex tasks."""

from __future__ import annotations

import os
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

WEAVIATE_COLLECTION = os.getenv("WEAVIATE_COLLECTION", "DrugLawDocs")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "jina").lower()
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "jina-embeddings-v3" if EMBEDDING_PROVIDER == "jina" else "text-embedding-3-small",
)
EMBEDDING_DIM = 1024 if EMBEDDING_PROVIDER == "jina" else 1536


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    placeholders = {"", "xxx", "sk-xxx", "jina_xxx", "pi_xxx", "https://xxx.weaviate.network"}
    if value in placeholders:
        raise RuntimeError(f"Missing production environment variable: {name}")
    return value


def get_openai_client():
    from openai import OpenAI

    return OpenAI(api_key=require_env("OPENAI_API_KEY"))


def embed_texts(texts: list[str], task: str = "retrieval.passage") -> list[list[float]]:
    if EMBEDDING_PROVIDER == "jina":
        import requests

        response = requests.post(
            "https://api.jina.ai/v1/embeddings",
            headers={
                "Authorization": f"Bearer {require_env('JINA_API_KEY')}",
                "Content-Type": "application/json",
            },
            json={"model": EMBEDDING_MODEL, "task": task, "input": texts},
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        return [item["embedding"] for item in payload["data"]]

    client = get_openai_client()
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def connect_weaviate():
    import weaviate
    from weaviate.classes.init import Auth

    url = os.getenv("WEAVIATE_URL", "").strip()
    api_key = os.getenv("WEAVIATE_API_KEY", "").strip()
    headers = {"X-OpenAI-Api-Key": require_env("OPENAI_API_KEY")}

    if not url or url == "https://xxx.weaviate.network":
        return weaviate.connect_to_embedded(
            hostname="127.0.0.1",
            port=int(os.getenv("WEAVIATE_EMBEDDED_PORT", "8079")),
            grpc_port=int(os.getenv("WEAVIATE_EMBEDDED_GRPC_PORT", "50050")),
            headers=headers,
            persistence_data_path=os.getenv("WEAVIATE_EMBEDDED_DATA_PATH", "data/weaviate-embedded"),
            environment_variables={
                "AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED": "true",
                "DEFAULT_VECTORIZER_MODULE": "none",
                "LOG_LEVEL": "error",
            },
        )

    if api_key and api_key != "xxx":
        return weaviate.connect_to_weaviate_cloud(
            cluster_url=url,
            auth_credentials=Auth.api_key(api_key),
            headers=headers,
        )

    parsed = urlparse(url)
    if not parsed.hostname:
        raise RuntimeError(f"Invalid WEAVIATE_URL: {url}")
    secure = parsed.scheme == "https"
    http_port = parsed.port or (443 if secure else 80)
    grpc_port = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
    return weaviate.connect_to_custom(
        http_host=parsed.hostname,
        http_port=http_port,
        http_secure=secure,
        grpc_host=parsed.hostname,
        grpc_port=grpc_port,
        grpc_secure=secure,
        headers=headers,
    )
