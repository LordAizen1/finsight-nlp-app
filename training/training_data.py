# training_data.py

# This is the format spaCy needs for training:
# ("Text of the sentence", {"entities": [(start_char, end_char, "LABEL")]})
# start_char is the index of the first character of the entity.
# end_char is the index of the first character AFTER the entity.

TRAIN_DATA = [
    # --- FIN_EVENT examples ---
    ("US stock market valuations at historic highs seen before great depression, dot-com crash.",
     {"entities": [(0, 2, "GPE"), (54, 72, "FIN_EVENT"), (74, 88, "FIN_EVENT")]}),

    ("The US stock market valuation has hit historic highs, with metrics like market-cap-to-GDP exceeding the Great Depression of 1929 and the dot-com crash in 2000.",
     {"entities": [(4, 6, "GPE"), (106, 134, "FIN_EVENT"), (139, 163, "FIN_EVENT")]}),

    ("For context, in 1999, the CAPE hit about 44 before the crash.",
     {"entities": [(18, 22, "DATE"), (53, 64, "FIN_EVENT")]}),

    ("The 2008 financial crisis wiped out trillions in market value across the globe.",
     {"entities": [(4, 28, "FIN_EVENT")]}),

    ("Many investors lost everything during the Black Monday crash of 1987.",
     {"entities": [(40, 68, "FIN_EVENT")]}),

    ("The housing bubble burst triggered a severe recession in the United States.",
     {"entities": [(4, 25, "FIN_EVENT"), (56, 74, "GPE")]}),

    ("Fears of a flash crash spooked traders on Wall Street last Tuesday.",
     {"entities": [(10, 21, "FIN_EVENT")]}),

    ("The Asian financial crisis of 1997 spread rapidly across emerging markets.",
     {"entities": [(4, 34, "FIN_EVENT")]}),

    ("The European debt crisis caused significant turmoil in the eurozone between 2010 and 2012.",
     {"entities": [(4, 25, "FIN_EVENT"), (76, 80, "DATE"), (85, 89, "DATE")]}),

    ("Bitcoin experienced a massive crypto crash in early 2022, losing over 60% of its value.",
     {"entities": [(33, 45, "FIN_EVENT"), (52, 56, "DATE")]}),

    # --- STOCK examples ---
    ("Tech giants like MSFT and IBM also saw gains.",
     {"entities": [(17, 21, "STOCK"), (26, 29, "STOCK")]}),

    ("Shares of AAPL surged 5% after the earnings report was released.",
     {"entities": [(10, 14, "STOCK")]}),

    ("TSLA dropped sharply following Elon Musk's latest comments on Twitter.",
     {"entities": [(0, 4, "STOCK"), (31, 40, "PERSON")]}),

    ("Investors are bullish on NVDA heading into the next quarter.",
     {"entities": [(24, 28, "STOCK")]}),

    ("AMZN and GOOG both reported record revenues this quarter.",
     {"entities": [(0, 4, "STOCK"), (9, 13, "STOCK")]}),

    ("The rally in NFLX pushed the stock above its 52-week high.",
     {"entities": [(13, 17, "STOCK")]}),

    ("JPM reported strong earnings, beating analyst expectations by a wide margin.",
     {"entities": [(0, 3, "STOCK")]}),

    # --- Mixed examples ---
    ("Warren Buffett warned about market excess before the dot-com bubble burst.",
     {"entities": [(0, 14, "PERSON"), (51, 73, "FIN_EVENT")]}),

    ("Goldman Sachs raised its price target on AAPL to $250 amid strong iPhone sales.",
     {"entities": [(0, 13, "ORG"), (41, 45, "STOCK")]}),

    ("The Federal Reserve cut interest rates in response to the 2008 recession.",
     {"entities": [(4, 19, "ORG"), (58, 73, "FIN_EVENT")]}),

    ("META shares climbed after Mark Zuckerberg announced a new AI strategy.",
     {"entities": [(0, 4, "STOCK"), (26, 41, "PERSON")]}),

    ("The S&P 500 fell sharply during the Covid crash of March 2020.",
     {"entities": [(35, 46, "FIN_EVENT"), (50, 60, "DATE")]}),

    ("BA and LMT rose on news of a major defense contract from the Pentagon.",
     {"entities": [(0, 2, "STOCK"), (7, 10, "STOCK"), (60, 69, "ORG")]}),
]
