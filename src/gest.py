import pandas as pd

gest = pd.read_csv('./data/gest.csv')

stereotype_names = [
    f'{name} #{i+1}'
    for i, name in enumerate("""
        Emotional
        Gentle
        Empathetic
        Neat
        Social
        Weak
        Beautiful
        Tough
        Self-confident
        Professional
        Rational
        Providers
        Leaders
        Childish
        Sexual
        Strong
        """.strip().split('\n')
    )
]