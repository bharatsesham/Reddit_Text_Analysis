interrogative_word_list = ['what', 'which', 'whose', 'when', 'where', 'who', 'whom', 'whose', 'why', 'how', 'whether']
interrogative_pos_tag = ['WDT', 'WP', 'WP$', 'WRB']  # Interrogative (also includes relative pronoun)

""" Imperative (commonly for advice, suggestions, requests, commands, orders or instructions)
        1. First Person - 
        2. Second Person
            2.1 Polite:
                Something <PRP$><NN> # Move (NNP) out of my way! or Clean (JJ) your room.
                
            2.2 Direct:
                <NNP|JJ|VB><DT>*<NN> # Pass (NNP) the salt. or Complete (JJ) these by tomorrow.
                # Be there at five.
        3. Third Person - """

imperative_chunkgram = r"""VB-Phrase: {<NNP|JJ|VB><DT>*<NN>}
                           VB-Phrase: {<VB><RB>}"""



                        # """VB-Phrase: {<VB><RB>} # Go away.
                        #    VB-Phrase: {<VB|NNP|MD>*<PRP.*.?>*<VB|NN|RB>} # Keep quiet.
                        #    VB-Phrase: {<NNP><NN|WDT>} # Have fun. / Stop that.
                        #    VB-Phrase: {<VB><NN>}"""


    # chunkgram = r"""VB-Phrase: {<DT><,>*<VB>}
    #                 VB-Phrase: {<RB><VB>}
    #                 VB-Phrase: {<UH><,>*<VB>}
    #                 VB-Phrase: {<UH><,><VBP>}
    #                 VB-Phrase: {<PRP><VB>}
    #                 VB-Phrase: {<NN.?>+<,>*<VB>}
    #                 VB-Phrase: {<NN><IN>*<NN>}
    #                 Q-Tag: {<,><MD><RB>*<PRP><.>*}"""




# https://stackoverflow.com/questions/29473169/approach-for-identifying-whether-a-sentence-includes-an-imperative-within-it