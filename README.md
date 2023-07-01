# Measuring Online Emotional Reactions to Offline Events

A method to detect, measure and explain collective emotional and moral reactions on social media in response to offline events. Here is a flow chart of our pipeline.

<img width="801" alt="fig_flowchart" src="https://github.com/fionasguo/emotional_reaction/assets/44278097/c0cc0c57-3119-4d2e-a917-62e8aab7fc9a">

## Data
Timestamped texts, e.g. tweets

LA data: raw data in `minds02.isi.edu:/lfs1/jeff_data`; processed data (emotion and MF machine labeled) in `/nas/home/myusername/LA_tweets_emot_mf`.

## Affect Detection
### Emotion Detection 
We utilized [SpanEmo](https://github.com/hasanhuz/SpanEmo) trained on [SemEval 2018 Task 1e-c](https://competitions.codalab.org/competitions/17751#learn_the_details-datasets). 

On Donut cluster, `/nas/home/myusername/SpanEmo`, python 3.6.9, see requirements_3_6_9.txt to setup environments. Run emot.slurm script.

### Moral Sentiment Detection
We use the transformer-based pretrained language model BERT, and train it with a large amount of data from different sources, including the Moral Foundation Twitter Corpus dataset, the dataset of political tweets published by US congress members, a manually annotated COVID dataset, and the Extended Moral Foundations Dictionary data. See code for training and test [here](https://github.com/fionasguo/DAMF).

On Donut cluster `/data/vast/users/myusername/DAMF`, python 3.9.13, see requirements_3_9_13.txt, run la_mf.sh script.

With emotions and moral foundations annotated on individual tweets, we aggregate them to construct the time series of daily tweet requency for each category. `donut-submit01:event_changes/data/LA_tweets_emot_mf_agg.csv`

## Change point detection on aggregate emotions and moral sentiments
Environment: python 3.9.17, see cpd_env_requirements_python_3_9_17.txt to setup (THIS ENV IS TRICKY, only successfully set up on local laptop). Run LA_agg_data_ts_plot_cpd.ipynb notebook (on COVID data: CPD_ts_from_ashwin.ipynb, this has measuring the magnitude of changes).

## Topic Modeling
python 3.9.13, setup with requirements_topic_modeling_3_9_13.txt, run topic_modeling_multi.py

