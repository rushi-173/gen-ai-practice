import tiktoken

encoder = tiktoken.encoding_for_model('gpt-4o')

print("Vocab Size - ", encoder.n_vocab) 

text = "Rushikesh knows AI"
tokens = encoder.encode(text)

print("Tokens - ", tokens)

my_tokens = [169704, 507, 8382, 13484, 20837]
decoded = encoder.decode(my_tokens)

print("Decoded - ", decoded)