# Dissertation
## Research Question: Can Chatbots Truly Be 'Unbiased'?

This project aims to answer the above question by subjecting multiple chatbots, trained over many different types of dataset, to the Implicit Association Test (IAT), which can be found [here](https://implicit.harvard.edu/implicit/). This repository shows the code used to load the datasets, train chatbots and test them, using an Textual User Interface (TUI).

*Note*: The original research question for this project was "Can Adding Bias to a Machine Make it more believable?". As part of this, a basic Flask server was coded in order to later facilitate the completion of online Turing Tests to test the believability of the chatbots made. However, the research question changed before I had figured out how to use Flask properly, and thus well before this server was completed. The code has been left, as an insight into how the code was planned to be structured, and how I was learning Flask, before the change.

Because of the change the only files of note are in `main.py` and the `chatbot` folder, all other files were discontinued for this project!

## Usage

First, download the repository:
```
git clone https://github.com/jopokemine/Dissertation.git
```

Next, you will need to download the datasets, which can be found under the datasets heading. They should be placed in the folder `chatbot/data`.

Once the datasets are installed into the datasets folder, run the following to train a chatbot:
```
python3 main.py -tr -d [datasets]
```
And the following to test a chatbot:
```
python3 main.py -te -d [datasets]
```

## Datasets
The available datasets, and where to get them, are:
- Amazon [link](https://github.com/PolyAI-LDN/conversational-datasets)
- - Credit: Henderson, M., Budzianowski, P., Casanueva, I., Coope, S., Gerz, D., Kumar,G., Mrkši ́c, N., Spithourakis, G., Su, P.-H., Vulic, I., & Wen, T.-H. (2019). A repository of conversational datasets [Data available at github.com/PolyAI-LDN/conversational-datasets]. Proceedings of the Workshop on NLP for Conversational AI. https://arxiv.org/abs/1904.06472. License: Apache License, Version 2.0.
- Convai [link](http://convai.io/2018/data/)
- - Credit: Aliannejadi, M., Kiseleva, J., Chuklin, A., Dalton, J., & Burtsev, M. (2020). Con-vAI3: Generating Clarifying Questions for Open-Domain Dialogue Systems (ClariQ). https://arxiv.org/abs/2009.11352.
- Cornell [link](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
- - Credit: Danescu-Niculescu-Mizil, C., & Lee, L. (2011). Chameleons in imagined conversations: A new approach to understanding coordination of linguisticstyle in dialogs.Proceedings of the Workshop on Cognitive Modelingand Computational Linguistics, ACL 2011.
- OpenSubtitles [link](https://github.com/PolyAI-LDN/conversational-datasets)
- - Credit: Henderson, M., Budzianowski, P., Casanueva, I., Coope, S., Gerz, D., Kumar,G., Mrkši ́c, N., Spithourakis, G., Su, P.-H., Vulic, I., & Wen, T.-H. (2019). A repository of conversational datasets [Data available at github.com/PolyAI-LDN/conversational-datasets]. Proceedings of the Workshop on NLP for Conversational AI. https://arxiv.org/abs/1904.06472. License: Apache License, Version 2.0.
- QA [link](http://www.cs.cmu.edu/~ark/QA-data/)
- - Credit: Smith, N. A., Heilman, M., & Hwa, R. (2008). Question generation as a competitive undergraduate course project. Proceedings of the NSF Workshopon the Question Generation Shared Task and Evaluation Challenge, 4–6.
- Reddit [link](https://github.com/PolyAI-LDN/conversational-datasets)
- - Credit: Credit: Henderson, M., Budzianowski, P., Casanueva, I., Coope, S., Gerz, D., Kumar,G., Mrkši ́c, N., Spithourakis, G., Su, P.-H., Vulic, I., & Wen, T.-H. (2019). A repository of conversational datasets [Data available at github.com/PolyAI-LDN/conversational-datasets]. Proceedings of the Workshop on NLP for Conversational AI. https://arxiv.org/abs/1904.06472. License: Apache License, Version 2.0.
- SQuAD [link](https://rajpurkar.github.io/SQuAD-explorer/)
- - Credit: Rajpurkar, P., Jia, R., & Liang, P. (2018). Know What You Don’t Know: Unanswerable Questions for SQuAD. CoRR,abs/1806.03822. https://arxiv.org/abs/1806.03822.
- Twitter [link](https://www.kaggle.com/thoughtvector/customer-support-on-twitter/version/10)
- - Credit: Axelbrooke, S. (2017).Customer Support on Twitter(Version 10). RetrievedJanuary 5, 2021, from https://www.kaggle.com/thoughtvector/customer-support-on-twitter/version/10. License: Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).

*Note*: Due to difficulties sensibly creating sentence pairs from the data available, the Reddit dataset remains unfinished!
