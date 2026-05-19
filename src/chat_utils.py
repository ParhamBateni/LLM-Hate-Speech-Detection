def build_chat_messages(
    tokenizer,
    system_prompt: str,
    user_text: str,
) -> str:
    """Build a prompt string for the model (chat template or plain text for T5)."""
    query = f'QUERY: "{user_text}"'
    ct = getattr(tokenizer, "chat_template", None)
    if ct is None:
        # T5/FLAN: drop PREDICTION format lines (prime a constant label); complete after Answer:
        rules = system_prompt.split("Respond ONLY in the following format:")[0].strip()
        return f"{rules}\n\nQUERY: \"{user_text}\"\n\nPREDICTION:"
    if isinstance(ct, str) and "System role not supported" in ct:
        return f"SYSTEM: {system_prompt.strip()}\n\nUSER: {query}"
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    return tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )
