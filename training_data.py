# training_data.py

# This is the format spaCy needs for training:
# ("Text of the sentence", {"entities": [(start_char, end_char, "LABEL")]})
# start_char is the index of the first character of the entity.
# end char is the index of the first character AFTER the entity.
TRAIN_DATA = [
    ("US stock market valuations at historic highs seen before great depression, dot-com crash.",
     {"entities": [(0, 2, "GPE"), (54, 72, "FIN_EVENT"), (74,
88, "FIN_EVENT")]}),

    ("The US stock market valuation has hit historic highs, with metrics like market-cap-to-GDP exceeding the Great Depression of 1929 and the dot-com crash in 2000.",
     {"entities": [(4, 6, "GPE"), (106, 134, "FIN_EVENT"), (
139, 163, "FIN_EVENT")]}),

    ("For context, in 1999, the CAPE hit about 44 before the crash.",
     {"entities": [(18, 22, "DATE"), (53, 64, "FIN_EVENT"
)]}),

    ("Tech giants like MSFT and IBM also saw gains.",
     {"entities": [(17, 21, "STOCK"), (26, 29, "STOCK")]})
]