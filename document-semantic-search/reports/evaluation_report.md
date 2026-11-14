# Chunking Strategy Evaluation Report

*Generated automatically on 2026-11-14 10:11:09 by running the implemented pipeline.*

## Setup

- Embedding backend used for this run: **`hashing:384d`**
- ⚠️ **This run used the offline hashing fallback embedder, not `all-MiniLM-L6-v2`.** The fallback matches words rather than meaning, so similarity scores and rankings are NOT representative of the real model. Re-run `python main.py --reset` on a machine with internet access to regenerate this report with MiniLM.
- Chunk size: 800 characters, overlap: 150 characters
- Results per strategy per query (top-k): 5
- Scores: **cosine similarity** = 1 − ChromaDB cosine distance (range −1..1, higher is better; not a percentage)
- Execution time: median of 3 runs, in milliseconds, covering query embedding + vector search
- *Relevant* = chunk mentions at least 3 of the query's expected keywords (automatic heuristic; read the retrieved text to judge for yourself)

## Overall summary (all queries)

| strategy | results | mean_similarity | top1_similarity | precision_at_k | mean_keyword_matches | distinct_documents | avg_chunk_chars | avg_time_ms |
|---|---|---|---|---|---|---|---|---|
| fixed | 15 | 0.277 | 0.401 | 0.733 | 3.533 | 3 | 768.0 | 3.357 |
| recursive | 15 | 0.298 | 0.483 | 0.733 | 3.133 | 3 | 583.0 | 3.747 |

## Query: What are the main benefits of artificial intelligence in healthcare?

| strategy | results | mean_similarity | top1_similarity | precision_at_k | mean_keyword_matches | distinct_documents | avg_chunk_chars | avg_time_ms |
|---|---|---|---|---|---|---|---|---|
| fixed | 5 | 0.212 | 0.296 | 0.8 | 3.6 | 2 | 704.0 | 3.33 |
| recursive | 5 | 0.226 | 0.32 | 0.8 | 3.8 | 2 | 559.0 | 3.38 |

**Automatic comparison:** mean similarity: **recursive** higher (fixed 0.212 vs recursive 0.226); precision@5: tie (0.8); keyword matches: **recursive** higher (fixed 3.6 vs recursive 3.8)

### fixed chunking: top 5 results

| rank | document | page | similarity | kw | time_ms | retrieved chunk (excerpt) |
|---|---|---|---|---|---|---|
| 1 | ai_in_healthcare.pdf | 2 | 0.2962 | 6 | 3.33 | t likely to work for a particular patient, a field known as personalised or precision medicine. In cancer care, for example, the genetic profile of a tumour can guide the choice of therapy. Wearable devices and home sensors stream heart rate, blood glucose, sleep and activity data. Algorithms can detect an irregular h… |
| 2 | ai_in_healthcare.pdf | 1 | 0.2442 | 5 | 3.33 | Artificial Intelligence in Healthcare: Benefits, Risks and Responsible Use A practical overview for students and technology teams 1. Why Healthcare Is Turning to Artificial Intelligence Health systems around the world face rising demand, ageing populations, shortages of trained staff and growing volumes of data. A sin… |
| 3 | data_security_and_privacy.pdf | 1 | 0.2249 | 1 | 3.33 | ity and availability triad. Confidentiality means only authorised people can read the data. Integrity means the data cannot be changed without detection. Availability means the data and the systems that process it can be reached when needed. Artificial intelligence systems add new ways to fail at each of these three g… |
| 4 | ai_in_healthcare.pdf | 1 | 0.156 | 3 | 3.33 | s. The term covers several techniques. Machine learning models find patterns in structured records, deep learning analyses images and signals, and natural language processing reads free-text notes. Most successful systems today are assistants rather than replacements: they prepare a recommendation and a qualified pers… |
| 5 | ai_in_healthcare.pdf | 4 | 0.1388 | 3 | 3.33 | Looking ahead, foundation models that combine text, images and laboratory values may support more general assistants, but the same principles apply: demonstrated benefit, fairness across patient groups, transparency and protection of privacy are the conditions for earning the trust on which healthcare ultimately depen… |

### recursive chunking: top 5 results

| rank | document | page | similarity | kw | time_ms | retrieved chunk (excerpt) |
|---|---|---|---|---|---|---|
| 1 | ai_in_healthcare.pdf | 2 | 0.32 | 7 | 3.38 | Wearable devices and home sensors stream heart rate, blood glucose, sleep and activity data. Algorithms can detect an irregular heart rhythm or a dangerous trend in blood sugar and notify the patient or the care team. Remote monitoring lets people with chronic illnesses stay at home, extends care to rural communities… |
| 2 | ai_in_healthcare.pdf | 1 | 0.3101 | 5 | 3.38 | Artificial Intelligence in Healthcare: Benefits, Risks and Responsible Use A practical overview for students and technology teams 1. Why Healthcare Is Turning to Artificial Intelligence Health systems around the world face rising demand, ageing populations, shortages of trained staff and growing volumes of data. A sin… |
| 3 | ai_in_healthcare.pdf | 1 | 0.1935 | 3 | 3.38 | 2. Medical Imaging and Diagnosis Medical imaging is the area where artificial intelligence has produced some of its clearest benefits. Convolutional neural networks trained on large collections of labelled scans can detect suspicious regions in chest radiographs, mammograms, retinal photographs and computed tomography… |
| 4 | data_security_and_privacy.pdf | 1 | 0.1669 | 1 | 3.38 | Security practice is often summarised by the confidentiality, integrity and availability triad. Confidentiality means only authorised people can read the data. Integrity means the data cannot be changed without detection. Availability means the data and the systems that process it can be reached when needed. Artificia… |
| 5 | ai_in_healthcare.pdf | 4 | 0.1388 | 3 | 3.38 | Looking ahead, foundation models that combine text, images and laboratory values may support more general assistants, but the same principles apply: demonstrated benefit, fairness across patient groups, transparency and protection of privacy are the conditions for earning the trust on which healthcare ultimately depen… |

## Query: How does machine learning use training data to make predictions?

| strategy | results | mean_similarity | top1_similarity | precision_at_k | mean_keyword_matches | distinct_documents | avg_chunk_chars | avg_time_ms |
|---|---|---|---|---|---|---|---|---|
| fixed | 5 | 0.338 | 0.373 | 1.0 | 5.0 | 2 | 799.0 | 3.4 |
| recursive | 5 | 0.315 | 0.373 | 1.0 | 3.6 | 2 | 649.0 | 3.56 |

**Automatic comparison:** mean similarity: **fixed** higher (fixed 0.338 vs recursive 0.315); precision@5: tie (1.0); keyword matches: **fixed** higher (fixed 5.0 vs recursive 3.6)

### fixed chunking: top 5 results

| rank | document | page | similarity | kw | time_ms | retrieved chunk (excerpt) |
|---|---|---|---|---|---|---|
| 1 | machine_learning_fundamentals.pdf | 1 | 0.3732 | 6 | 3.4 | ining certain words should go to a junk folder. This approach breaks down when the rules are too numerous, too subtle or constantly changing. Machine learning replaces the hand-written rules with a training procedure. The programmer supplies examples of emails that people marked as junk and emails they kept, and the l… |
| 2 | machine_learning_fundamentals.pdf | 1 | 0.3458 | 4 | 3.4 | Machine Learning Fundamentals: Data, Training and Prediction An introductory guide for software engineering students 1. Introduction Machine learning is a branch of artificial intelligence in which a computer program improves its behaviour by studying examples instead of following rules written by hand. A spam filter,… |
| 3 | machine_learning_fundamentals.pdf | 2 | 0.3437 | 7 | 3.4 | uring development leaks information and produces over-optimistic results. Bias in the training data becomes bias in the predictions. If one group of people is under-represented, the model will usually perform worse for that group. Careful sampling, documentation of where the data came from and evaluation on separate s… |
| 4 | machine_learning_fundamentals.pdf | 2 | 0.3213 | 5 | 3.4 | features that make patterns easier to learn, for instance converting a timestamp into day of week and hour. Deep neural networks reduce the amount of manual feature engineering because they learn useful representations directly from raw inputs, but they need far more data to do so. To judge how well a model will perfo… |
| 5 | data_security_and_privacy.pdf | 1 | 0.3056 | 3 | 3.4 | harm of a leak. The fifth is regulatory complexity, since companies that serve customers in many countries must satisfy several overlapping laws at the same time. 3. Threats Specific to Machine Learning Systems Data poisoning occurs when an attacker inserts misleading examples into the training data so that the finish… |

### recursive chunking: top 5 results

| rank | document | page | similarity | kw | time_ms | retrieved chunk (excerpt) |
|---|---|---|---|---|---|---|
| 1 | machine_learning_fundamentals.pdf | 1 | 0.3731 | 4 | 3.56 | Traditional software is built from explicit instructions. A programmer decides that an email containing certain words should go to a junk folder. This approach breaks down when the rules are too numerous, too subtle or constantly changing. Machine learning replaces the hand-written rules with a training procedure. The… |
| 2 | machine_learning_fundamentals.pdf | 1 | 0.33 | 4 | 3.56 | Machine Learning Fundamentals: Data, Training and Prediction An introductory guide for software engineering students 1. Introduction Machine learning is a branch of artificial intelligence in which a computer program improves its behaviour by studying examples instead of following rules written by hand. A spam filter,… |
| 3 | machine_learning_fundamentals.pdf | 2 | 0.2922 | 4 | 3.56 | To judge how well a model will perform in the real world, the dataset is divided into separate parts. The training set is used to fit the model. A validation set is used to compare settings and choose between candidate models. A test set is kept untouched until the very end and gives an honest estimate of performance… |
| 4 | machine_learning_fundamentals.pdf | 1 | 0.2911 | 3 | 3.56 | 2. Types of Machine Learning Supervised learning uses labelled examples. Every training record contains input features together with the correct answer, which is called the label or target. When the label is a category, such as fraud or not fraud, the task is classification. When the label is a number, such as the pri… |
| 5 | data_security_and_privacy.pdf | 1 | 0.2876 | 3 | 3.56 | 3. Threats Specific to Machine Learning Systems Data poisoning occurs when an attacker inserts misleading examples into the training data so that the finished model behaves badly, for example by misclassifying a particular input. Systems that learn from user submissions or from data scraped from the internet are espec… |

## Query: What are the main challenges of data security and privacy?

| strategy | results | mean_similarity | top1_similarity | precision_at_k | mean_keyword_matches | distinct_documents | avg_chunk_chars | avg_time_ms |
|---|---|---|---|---|---|---|---|---|
| fixed | 5 | 0.282 | 0.401 | 0.4 | 2.0 | 2 | 800.0 | 3.34 |
| recursive | 5 | 0.354 | 0.483 | 0.4 | 2.0 | 2 | 541.0 | 4.3 |

**Automatic comparison:** mean similarity: **recursive** higher (fixed 0.282 vs recursive 0.354); precision@5: tie (0.4); keyword matches: tie (2.0)

### fixed chunking: top 5 results

| rank | document | page | similarity | kw | time_ms | retrieved chunk (excerpt) |
|---|---|---|---|---|---|---|
| 1 | data_security_and_privacy.pdf | 1 | 0.4012 | 2 | 3.34 | ity and availability triad. Confidentiality means only authorised people can read the data. Integrity means the data cannot be changed without detection. Availability means the data and the systems that process it can be reached when needed. Artificial intelligence systems add new ways to fail at each of these three g… |
| 2 | data_security_and_privacy.pdf | 1 | 0.3849 | 3 | 3.34 | Data Security and Privacy in the Age of AI Principles, threats and safeguards for data-driven systems 1. Security and Privacy: Two Related Goals Data security and data privacy are closely connected but not identical. Security is about protecting information from unauthorised access, alteration or destruction. Privacy… |
| 3 | data_security_and_privacy.pdf | 2 | 0.2103 | 3 | 3.34 | val of accounts when people change roles shrink the damage a single stolen password can cause. Detailed audit logs record who accessed which records, which allows suspicious behaviour to be detected and investigated. Secure development and operations complete the picture: patching known vulnerabilities promptly, scann… |
| 4 | ai_in_healthcare.pdf | 3 | 0.2082 | 0 | 3.34 | resources. These gains are real but are not automatic. They depend on high-quality data, careful integration into clinical workflows and continuous evaluation. 8. Challenges and Risks Data quality and bias are the most serious technical challenges. A model trained mostly on patients from one country, age group or ethn… |
| 5 | data_security_and_privacy.pdf | 2 | 0.2071 | 2 | 3.34 | nisation should collect only what is necessary for a stated purpose. Purpose limitation says data gathered for one reason should not quietly be reused for an unrelated one. Transparency requires clear notices that explain what is done with personal information, and consent, where it is the legal basis, must be informe… |

### recursive chunking: top 5 results

| rank | document | page | similarity | kw | time_ms | retrieved chunk (excerpt) |
|---|---|---|---|---|---|---|
| 1 | data_security_and_privacy.pdf | 1 | 0.4826 | 3 | 4.3 | 2. The Main Challenges of Data Security and Privacy The main challenges of data security and privacy can be grouped into a few themes. The first is scale: organisations now collect and copy data across cloud services, laptops, partners and analytics platforms, so it is hard to know where every sensitive record lives.… |
| 2 | data_security_and_privacy.pdf | 1 | 0.4193 | 3 | 4.3 | Data Security and Privacy in the Age of AI Principles, threats and safeguards for data-driven systems 1. Security and Privacy: Two Related Goals Data security and data privacy are closely connected but not identical. Security is about protecting information from unauthorised access, alteration or destruction. Privacy… |
| 3 | data_security_and_privacy.pdf | 1 | 0.3685 | 2 | 4.3 | Security practice is often summarised by the confidentiality, integrity and availability triad. Confidentiality means only authorised people can read the data. Integrity means the data cannot be changed without detection. Availability means the data and the systems that process it can be reached when needed. Artificia… |
| 4 | data_security_and_privacy.pdf | 2 | 0.272 | 2 | 4.3 | Storage limitation requires that data be deleted once it is no longer needed. Individuals are granted rights to access their data, to have errors corrected and, in many jurisdictions, to have their data erased. Privacy by design asks teams to build these protections into a system from the start instead of adding them… |
| 5 | ai_in_healthcare.pdf | 3 | 0.2264 | 0 | 4.3 | resources. These gains are real but are not automatic. They depend on high-quality data, careful integration into clinical workflows and continuous evaluation. 8. Challenges and Risks Data quality and bias are the most serious technical challenges. A model trained mostly on patients from one country, age group or ethn… |

## Notes and limitations

- The CSV file `chunking_comparison.csv` contains the full text of every retrieved chunk.
- Keyword-based relevance is only a proxy; with three queries and three documents this is a small experiment, not a benchmark.
- Small differences in timing are within normal run-to-run noise on a laptop.
