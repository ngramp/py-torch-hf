import gensim
from gensim import corpora
from textblob import TextBlob
from nltk.corpus import stopwords
import nltk
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

nltk.download('stopwords')

# Sample text extracted from the webpage
webpage_text = ("Five mums have died after family courts allowed fathers accused of abuse to apply for contact with "
                "their children, a BBC investigation has found. Some took their own lives, another had a heart attack "
                "outside court. One mother, whose child was ordered to live with a convicted child rapist, "
                "would no longer eat and drink and gave up living, friends say. A separate study has found 75 "
                "children forced into contact with fathers who had been previously reported for abuse. In some cases, "
                "in the England-wide study carried out by the University of Manchester and revealed for the first "
                "time by the BBC, the fathers were convicted paedophiles. All the fathers in the study had responded "
                "in court to allegations of abuse with a disputed concept known as parental alienation, in which they "
                "claimed the mothers had turned the child against them without good reason. Dr Elizabeth Dalgarno, "
                "who led the research, says the concept is a handy tool for abusers and its acceptance by courts is a "
                "national scandal. Family law barrister, Lucy Reed KC, says the term is deployed increasingly "
                "frequently - but doesn't always mean the same thing. It's quite often used by fathers to mean pretty "
                "much anything that is in opposition to their demand for a certain amount of contact. The 45 mothers "
                "of the children in the University of Manchester study all reported serious health problems which "
                "they believed were linked to the stress of family court proceedings - including miscarriages, "
                "heart attacks and suicidal thoughts. For months, the BBC has also been examining stories of "
                "traumatised women as part of a wider investigation into the way the family courts handle domestic "
                "violence claims in disputes between parents. Because of the laws surrounding reporting of the court "
                "proceedings, intended to protect children, the womens names and some identifying details have been "
                "changed.")

# Preprocess the text (tokenization, lowercasing, etc.)

# Tokenize the preprocessed text
tokens = [word for word in webpage_text.split()]

stop_words = set(stopwords.words("english"))  # Adjust the language if needed
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

# Create a Gensim dictionary from the tokens
dictionary = corpora.Dictionary([filtered_tokens])

# Create a Gensim corpus
corpus = [dictionary.doc2bow(filtered_tokens)]

# Perform topic modeling (e.g., Latent Dirichlet Allocation - LDA)
lda_model = gensim.models.LdaModel(corpus, num_topics=10, id2word=dictionary)

# Extract the topics
topics = lda_model.print_topics(num_words=15)

# Analyze sentiment for each topic
topic_sentiments = {}
for topic_id, topic in topics:
    blob = TextBlob(topic)
    sentiment_score = blob.sentiment.polarity
    topic_sentiments[topic_id] = sentiment_score

# Categorize topics based on sentiment scores (adjust thresholds as needed)
positive_topics = [topic_id for topic_id, score in topic_sentiments.items() if score > 0.2]
negative_topics = [topic_id for topic_id, score in topic_sentiments.items() if score < -0.2]
neutral_topics = [topic_id for topic_id, score in topic_sentiments.items() if -0.2 <= score <= 0.2]

# Print or visualize the categorized topics and their sentiment
print("Positive Topics:", positive_topics)
print("Negative Topics:", negative_topics)
print("Neutral Topics:", neutral_topics)

for topic in topics:
    print("topic:", topic)

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

input_ids = tokenizer.encode(
    "summarize: " + webpage_text,
    return_tensors="pt",
    max_length=1024,
    truncation=True
)
summary_ids = model.generate(
    input_ids,
    max_length=150,
    min_length=40,
    length_penalty=2.0,
    num_beams=4,
    early_stopping=True
)

# Decode the summary tokens into text
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("Generated Summary:")
print(summary)
