
import os
from langchain_groq import ChatGroq

GENERATOR_MODEL = "openai/gpt-oss-20b"  # abhi Groq free-tier par accessible model

SYSTEM_PROMPT = """Tum NovaTech Solutions ke internal AI assistant ho. Sirf neeche diye
gaye context (company documents ke chunks) ke bunyad par sawal ka jawab do.

Rules:
- Sirf context mein maujood information use karo, bahar se kuch mat milao (hallucinate mat karo)
- Sawal ke exact alfaz context mein na milein, iska matlab jawab mojood nahi — pehle dhyan se
  dekho ke context mein wahi baat kisi aur tarike se (paraphrase, synonym, related term) to
  nahi likhi. "Ye information mojood nahi hai" sirf tab bolo jab poori tarah confirm ho jaye
  ke context mein related topic bhi nahi hai
- Agar jawab mein 2 ya zyada cheezein (log, vendors, tiers, rules) list kar rahe ho, to har
  item ka role/label bhi zaroor likho — sirf naam ek sath mat likho. Format clear aur
  unambiguous rakho (masalan: "X — [role], Y — [role], Z — [role]"), taake koi confusion na
  ho ke kaunsi cheez kis se related hai
- Agar context mein jawab na ho, saaf keh do "Ye information provided context mein maujood nahi hai"
- Jawab Roman Urdu mein, mukhtasir aur seedha do
"""


def build_generator():
    return ChatGroq(
        model=GENERATOR_MODEL,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
        max_tokens=1024,
        reasoning_format="hidden",
        reasoning_effort="low",
    )


def generate_answer(llm, question: str, retrieved_chunks: list[str]) -> str:
    context = "\n\n".join(f"[Chunk {i+1}]\n{c}" for i, c in enumerate(retrieved_chunks))
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nSawal: {question}\n\nJawab:"
    response = llm.invoke(prompt)
    return response.content.strip()


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    from retriever import Retriever

    retriever = Retriever(top_k=3)
    llm = build_generator()

    q = "Tier 3 Confidential data ko training mein kaise use kiya ja sakta hai?"
    chunks = retriever.retrieve(q)
    answer = generate_answer(llm, q, chunks)

    print("Question:", q)
    print("\nGenerated Answer:\n", answer)