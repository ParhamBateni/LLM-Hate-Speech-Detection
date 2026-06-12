def build_prompt(
    tokenizer,
    system_prompt: str,
    user_text: str,
) -> str:
    """Build a prompt string for the model (chat template or plain text fallback)."""
    ct = getattr(tokenizer, "chat_template", None)
    user_text = user_text.replace('"', "'")
    query = f'QUERY: "{user_text}"'

    if ct is None:
        return f"{system_prompt.strip()}\n\n{query}"

    # Gemma (and similar) reject a separate system role; fold instructions into the user turn.
    if isinstance(ct, str) and "System role not supported" in ct:
        conversation = [
            {"role": "user", "content": f"{system_prompt.strip()}\n\n{query}"},
        ]
    else:
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

    return tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )
