SYSTEM = """You answer questions using ONLY the numbered context passages provided.
Rules:
- Cite passages inline like [1] or [2][3] after each claim.
- If the context does not contain the answer, reply exactly: I don't know based on the provided documents.
- Never use outside knowledge. Be concise."""


def build_user_prompt(question: str, hits: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i}] ({h['source']}, p.{h['page']}, {h['heading']})\n{h['text']}"
        for i, h in enumerate(hits, 1)
    )
    return f"Context:\n{context}\n\nQuestion: {question}"
