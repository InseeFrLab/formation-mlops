"""
Preprocessor.
"""
import pandas as pd
import string
from nltk.corpus import stopwords as ntlk_stopwords
from nltk.stem.snowball import SnowballStemmer
import unidecode


class Preprocessor():
    """
    Preprocessor class.
    """

    def __init__(self):
        """
        Construct a Preprocessor object.
        """
        self.stopwords = tuple(ntlk_stopwords.words("french")) + \
            tuple(string.ascii_lowercase)
        self.stemmer = SnowballStemmer(language="french")

    def clean_text(self, df: pd.DataFrame, text_feature: str) -> pd.DataFrame:
        """
        Cleans a text feature for pd.DataFrame `df` at index idx.

        Args:
            df (pd.DataFrame): DataFrame.
            text_feature (str): Name of the text feature.

        Returns:
            df (pd.DataFrame): Clean DataFrame.
        """
        df = df.copy()

        # Fix encoding
        df[text_feature] = df[text_feature].map(unidecode.unidecode)

        # To lowercase
        df[text_feature] = df[text_feature].str.lower()

        # define replacement patterns
        replacements = {
            # Remove punctuations
            r"[^\w\s]": " ",
            # Remove numbers
            r"[\d+]": " ",
        }

        # apply replacements to text_feature column
        for pattern, replacement in replacements.items():
            df[text_feature] = df[text_feature].str.replace(
                pattern, replacement, regex=True
            )

        # Remove one-letter words
        df[text_feature] = df[text_feature].apply(
            lambda x: " ".join([w for w in x.split() if len(w) > 1])
        )

        # Tokenize texts
        libs_token = [lib.split() for lib in df[text_feature].to_list()]

        # Remove stopwords and stem
        df[text_feature] = [
            " ".join(
                [
                    self.stemmer.stem(word)
                    for word in libs_token[i]
                    if word not in self.stopwords
                ]
            )
            for i in range(len(libs_token))
        ]

        return df
