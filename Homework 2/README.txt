Austin Harris

Files:
    tagger.py - file containing project code
    output.txt - output of program over test sentences

Running:
    python3 tagger.py

Importing:
    from tagger import Tagger
    
Functions:
    Tagger.load_corpus(<Path to corput>) -> List of sentences 
    Tagger.initialize_probabilities(<list of sentences>) -> None
    Tagger.viterbi(<Sentence to parse>) -> List of part of speech tags
