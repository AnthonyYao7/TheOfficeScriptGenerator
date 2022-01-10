import numpy as np
import tensorflow as tf
from os import walk

"""
Steps:

1. Open every file in transcripts.
2. Read them into Ragged Tensors
3. Find the number of unique characters. 
4. Convert them into tokens with StringLookup
Done

5. Make training samples


"""

seq_length = 100


def text_from_ids(ids, table):
    return tf.strings.reduce_join(table(ids), axis=-1)


def read_transcripts():
    filenames = next(walk("transcripts"), (None, None, []))[2]

    scripts = []
    for file in filenames:
        with open(f"transcripts/{file}", "rb") as f:
            text = f.read().decode(encoding="utf-8")
            scripts.append(text)

    vocab = sorted(set(''.join(scripts)))

    chars = tf.strings.unicode_split(scripts, input_encoding="UTF-8")

    ids_from_chars = tf.keras.layers.StringLookup(
        vocabulary=list(vocab), mask_token=None)
    chars_from_ids = tf.keras.layers.StringLookup(
        vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

    tokenized = ids_from_chars(chars)

    return tokenized, ids_from_chars, chars_from_ids


def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


def process_single_episode(tokens, chars_from_ids):
    dataset = tf.data.Dataset.from_tensor_slices(tokens)

    examples_per_epoch = tokens.shape[0]

    sequences = dataset.batch(seq_length + 1, drop_remainder=True)



def main():
    tokens, token_table, char_table = read_transcripts()

    for i in range(tokens.shape[0]):
        process_single_episode(tokens[i], char_table)
        exit(69)



if __name__ == "__main__":
    main()
