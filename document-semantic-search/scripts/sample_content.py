"""Original educational text used to build the three sample PDFs.

These documents were written for this project (by Claude, an AI assistant made by
Anthropic). They are NOT copied from any external publication. The sandbox used to
build this project had no access to public document hosts, so original, clearly
labelled material was authored instead. See data/documents/SOURCES.md.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

Section = Tuple[str, List[str]]

DOCUMENTS: Dict[str, Dict[str, object]] = {}

# ---------------------------------------------------------------------------
# Document 1: Machine learning fundamentals
# ---------------------------------------------------------------------------
DOCUMENTS["machine_learning_fundamentals"] = {
    "title": "Machine Learning Fundamentals: Data, Training and Prediction",
    "subtitle": "An introductory guide for software engineering students",
    "sections": [
        (
            "1. Introduction",
            [
                "Machine learning is a branch of artificial intelligence in which a computer "
                "program improves its behaviour by studying examples instead of following "
                "rules written by hand. A spam filter, a movie recommender and a system that "
                "reads handwritten digits all rely on the same core idea: a mathematical "
                "model is fitted to data, and the fitted model is then used to make "
                "decisions about new situations it has never seen.",
                "Traditional software is built from explicit instructions. A programmer "
                "decides that an email containing certain words should go to a junk folder. "
                "This approach breaks down when the rules are too numerous, too subtle or "
                "constantly changing. Machine learning replaces the hand-written rules with "
                "a training procedure. The programmer supplies examples of emails that people "
                "marked as junk and emails they kept, and the learning algorithm discovers "
                "which patterns separate the two groups.",
                "This guide explains the main ingredients of a machine learning system: the "
                "kinds of learning, the role of training data, the way a model is trained, "
                "how predictions are produced, how mistakes such as overfitting appear, and "
                "how results are measured and monitored after deployment.",
            ],
        ),
        (
            "2. Types of Machine Learning",
            [
                "Supervised learning uses labelled examples. Every training record contains "
                "input features together with the correct answer, which is called the label "
                "or target. When the label is a category, such as fraud or not fraud, the "
                "task is classification. When the label is a number, such as the price of a "
                "house, the task is regression. Most business applications of machine "
                "learning today are supervised because the goal is usually well defined.",
                "Unsupervised learning works with data that has no labels. The algorithm "
                "searches for structure on its own. Clustering groups similar customers "
                "together, dimensionality reduction compresses many columns into a few "
                "informative ones, and anomaly detection flags records that look unlike the "
                "rest. Unsupervised methods are useful for exploring data and for building "
                "features that later supervised models can use.",
                "Reinforcement learning trains an agent that interacts with an environment. "
                "The agent takes actions, receives rewards or penalties, and gradually learns "
                "a strategy that maximises long-term reward. It has been used for game "
                "playing, robot control and resource scheduling. Between these families "
                "sits self-supervised learning, where the labels are created automatically "
                "from the data itself, for example by hiding a word in a sentence and asking "
                "the model to predict it. Modern language models are trained this way.",
            ],
        ),
        (
            "3. Training Data",
            [
                "The quality of a model is limited by the quality of its training data. "
                "Data should be relevant to the problem, large enough to represent the "
                "variety of real situations, and accurate. Missing values, duplicated rows, "
                "inconsistent units and wrongly labelled examples all teach the model the "
                "wrong lessons. Data scientists therefore spend a large part of every project "
                "cleaning and inspecting the dataset before any training begins.",
                "Each record is described by features, which are measurable properties such "
                "as age, number of purchases or the pixel values of an image. Feature "
                "engineering is the craft of turning raw information into features that make "
                "patterns easier to learn, for instance converting a timestamp into day of "
                "week and hour. Deep neural networks reduce the amount of manual feature "
                "engineering because they learn useful representations directly from raw "
                "inputs, but they need far more data to do so.",
                "To judge how well a model will perform in the real world, the dataset is "
                "divided into separate parts. The training set is used to fit the model. A "
                "validation set is used to compare settings and choose between candidate "
                "models. A test set is kept untouched until the very end and gives an honest "
                "estimate of performance on unseen data. Using the test set repeatedly during "
                "development leaks information and produces over-optimistic results.",
                "Bias in the training data becomes bias in the predictions. If one group of "
                "people is under-represented, the model will usually perform worse for that "
                "group. Careful sampling, documentation of where the data came from and "
                "evaluation on separate subgroups are the standard defences.",
            ],
        ),
        (
            "4. How a Model Learns",
            [
                "A model is a function with adjustable numbers called parameters. A linear "
                "model has one parameter per feature plus an intercept, while a large neural "
                "network can have billions. Training means finding parameter values that make "
                "the model's outputs agree with the labels in the training data.",
                "Agreement is measured by a loss function. For regression the loss is often "
                "the average squared difference between the prediction and the true value. For "
                "classification it is commonly the cross-entropy, which penalises confident "
                "wrong answers heavily. A smaller loss means a better fit.",
                "The most common optimisation method is gradient descent. The algorithm "
                "computes the gradient, which points in the direction in which the loss "
                "increases fastest, and then moves each parameter a small step in the opposite "
                "direction. The size of the step is the learning rate. If it is too large, "
                "training oscillates or diverges; if it is too small, training is painfully "
                "slow. In practice data is processed in small groups called mini-batches, and "
                "one complete pass through the training set is called an epoch. Training "
                "usually continues for many epochs until the validation loss stops improving.",
            ],
        ),
        (
            "5. Making Predictions",
            [
                "Once training has finished, the parameters are frozen and the model can be "
                "used for inference. A new record is converted into the same features that "
                "were used during training and passed through the model, which produces an "
                "output. The model does not look up an answer in the training data. It applies "
                "the patterns encoded in its parameters, which is why a trained model can "
                "generalise to inputs that were never part of the training set.",
                "Classification models usually output a probability for each class. A decision "
                "threshold turns the probability into a final label. A threshold of 0.5 is a "
                "common default, but it can be moved. A hospital screening tool might use a "
                "low threshold so that few real cases are missed, accepting more false alarms. "
                "A system that blocks financial transactions might use a high threshold to "
                "avoid inconveniencing honest customers.",
                "Predictions must be produced with exactly the same preprocessing that was "
                "applied during training. A frequent cause of failure in production is a "
                "mismatch between the two, such as a feature being scaled differently or a "
                "category appearing that the model has never seen. Packaging the preprocessing "
                "steps together with the model in one pipeline prevents this class of bugs.",
            ],
        ),
        (
            "6. Overfitting and Underfitting",
            [
                "Overfitting happens when a model memorises the quirks and noise of the "
                "training set instead of the real underlying pattern. It scores very well on "
                "the training data but poorly on new data. A telltale sign is a training loss "
                "that keeps falling while the validation loss starts to rise.",
                "Underfitting is the opposite problem. The model is too simple, or trained for "
                "too little time, to capture the pattern at all, so it performs badly on both "
                "the training and the validation data. Choosing a model of suitable "
                "complexity is called managing the bias-variance trade-off.",
                "Several techniques reduce overfitting. Collecting more data is the most "
                "reliable. Regularisation adds a penalty for large parameter values and "
                "encourages simpler models. Dropout randomly disables neurons during training "
                "so that no single neuron can be relied on too much. Early stopping halts "
                "training when validation performance stops improving. Cross-validation "
                "repeats the train and validate procedure on several different splits to "
                "obtain a more stable estimate of how well the model generalises.",
            ],
        ),
        (
            "7. Evaluating a Model",
            [
                "Accuracy, the fraction of predictions that are correct, is easy to "
                "understand but can be misleading. If only one transaction in a thousand is "
                "fraudulent, a model that always answers not fraud is 99.9 percent accurate "
                "and completely useless. For imbalanced problems other measures are needed.",
                "Precision is the share of positive predictions that were truly positive. "
                "Recall is the share of truly positive cases that the model found. The F1 "
                "score combines them into a single number using the harmonic mean. A confusion "
                "matrix lays out true positives, false positives, true negatives and false "
                "negatives so that the kinds of mistakes can be studied directly. For "
                "regression, typical measures are mean absolute error and root mean squared "
                "error, which are expressed in the units of the target.",
                "Metrics should always be chosen to reflect the real cost of mistakes. A "
                "missed cancer diagnosis and a wrongly flagged email do not carry the same "
                "consequences, so they should not be weighted equally. Performance should "
                "also be reported for important subgroups, because a good overall average can "
                "hide a serious failure on a minority of the users.",
            ],
        ),
        (
            "8. Common Algorithms",
            [
                "Linear regression and logistic regression are simple, fast and easy to "
                "interpret. They are strong baselines and should usually be the first models "
                "tried. Decision trees split the data using a sequence of questions about the "
                "features and are easy to visualise, but a single deep tree overfits readily.",
                "Ensemble methods combine many weak models into a strong one. Random forests "
                "average the votes of many trees trained on random subsets of the data. "
                "Gradient boosting builds trees one after another, each one correcting the "
                "errors of the previous ones, and is often the best choice for tabular data. "
                "Support vector machines and k-nearest neighbours are classic alternatives that "
                "still appear in some applications.",
                "Neural networks stack layers of simple units whose connections are learned. "
                "Convolutional networks excel at images, recurrent and transformer networks "
                "excel at sequences such as text and speech. They achieve the highest accuracy "
                "on perception and language problems, but require large datasets, specialised "
                "hardware and careful tuning, and their decisions are harder to explain.",
            ],
        ),
        (
            "9. Deployment and Monitoring",
            [
                "Training a model is only the beginning of its life. After deployment the "
                "world keeps changing, and a model that was accurate last year may degrade "
                "silently. Data drift occurs when the distribution of the incoming features "
                "shifts, for example when customers start using a new device. Concept drift "
                "occurs when the relationship between features and the correct answer itself "
                "changes, as when fraudsters adopt new tactics.",
                "Production systems therefore log inputs and predictions, track quality "
                "metrics over time, and raise alerts when they leave an agreed range. Models "
                "are retrained on fresh data on a schedule or when drift is detected, and "
                "every version is stored with the data and code that produced it so that "
                "results can be reproduced and, if necessary, rolled back.",
            ],
        ),
        (
            "10. Text Embeddings and Semantic Similarity",
            [
                "A text embedding model converts a sentence or paragraph into a vector of "
                "numbers, typically a few hundred values long. Texts with similar meaning are "
                "placed close together in this vector space, even when they share few words. "
                "The sentence about reducing the cost of a hospital stay and the sentence "
                "about lowering healthcare expenses end up near each other.",
                "Closeness is usually measured with cosine similarity, which compares the "
                "angle between two vectors. This property powers semantic search: documents "
                "are split into chunks, every chunk is embedded and stored in a vector "
                "database, and a user's question is embedded with the same model so that the "
                "nearest chunks can be retrieved quickly. The quality of the retrieved "
                "passages depends strongly on how the original text was chunked, which is why "
                "comparing chunking strategies is a worthwhile experiment.",
            ],
        ),
    ],
}

# ---------------------------------------------------------------------------
# Document 2: AI in healthcare
# ---------------------------------------------------------------------------
DOCUMENTS["ai_in_healthcare"] = {
    "title": "Artificial Intelligence in Healthcare: Benefits, Risks and Responsible Use",
    "subtitle": "A practical overview for students and technology teams",
    "sections": [
        (
            "1. Why Healthcare Is Turning to Artificial Intelligence",
            [
                "Health systems around the world face rising demand, ageing populations, "
                "shortages of trained staff and growing volumes of data. A single hospital "
                "produces medical images, laboratory results, clinical notes, prescriptions "
                "and sensor readings every day, far more than any clinician can review "
                "thoroughly. Artificial intelligence offers tools that can sift through this "
                "information, highlight what matters and free up professionals to spend more "
                "time with patients.",
                "The term covers several techniques. Machine learning models find patterns in "
                "structured records, deep learning analyses images and signals, and natural "
                "language processing reads free-text notes. Most successful systems today are "
                "assistants rather than replacements: they prepare a recommendation and a "
                "qualified person makes the final decision.",
            ],
        ),
        (
            "2. Medical Imaging and Diagnosis",
            [
                "Medical imaging is the area where artificial intelligence has produced some "
                "of its clearest benefits. Convolutional neural networks trained on large "
                "collections of labelled scans can detect suspicious regions in chest "
                "radiographs, mammograms, retinal photographs and computed tomography images. "
                "In screening programmes the software can act as a second reader, drawing the "
                "radiologist's attention to a subtle finding that might otherwise be missed.",
                "Automated triage is another valuable use. When hundreds of scans arrive in "
                "an emergency department, an algorithm can flag the ones that appear to show "
                "a stroke or a collapsed lung so that they are read first. Faster reading "
                "of the most urgent cases can shorten the time to treatment, which matters "
                "greatly for conditions in which every minute counts.",
                "Pathology and dermatology follow the same pattern. Models can count cells "
                "on digitised tissue slides or classify photographs of skin lesions, giving "
                "clinicians in areas with few specialists a way to obtain expert-level "
                "guidance. Evidence shows, however, that performance measured on a research "
                "dataset often drops when the model is used with a different scanner or "
                "patient population, so local validation is essential before use.",
            ],
        ),
        (
            "3. Clinical Decision Support and Early Warning",
            [
                "Clinical decision support systems combine patient data with medical "
                "knowledge to assist with diagnosis and treatment choices. Predictive models "
                "can scan the electronic health record continuously and estimate the risk "
                "that a hospitalised patient will develop sepsis, kidney injury or cardiac "
                "arrest in the coming hours. Early warning allows nurses and doctors to "
                "intervene before the patient deteriorates, which can reduce intensive care "
                "admissions and save lives.",
                "Such alerts must be designed with care. A system that produces too many "
                "false alarms causes alert fatigue, and staff begin to ignore it. Successful "
                "deployments tune thresholds with clinicians, explain which factors drove a "
                "score and measure the effect on patient outcomes, not merely the accuracy of "
                "the model in the laboratory.",
            ],
        ),
        (
            "4. Drug Discovery and Research",
            [
                "Developing a new medicine traditionally takes more than a decade and costs "
                "billions. Machine learning shortens early stages of the process. Models "
                "predict how a candidate molecule will bind to a target protein, estimate "
                "its toxicity and suggest chemical modifications, allowing researchers to "
                "test far fewer compounds in the laboratory. Advances in predicting the "
                "three-dimensional shape of proteins from their amino acid sequence have "
                "given biologists a powerful new starting point for designing treatments.",
                "In clinical research, algorithms help match patients to suitable trials by "
                "reading eligibility criteria and medical records, and they can monitor "
                "incoming trial data for safety signals. These applications do not replace "
                "experiments or regulatory review, but they can make research faster and "
                "better targeted.",
            ],
        ),
        (
            "5. Reducing Administrative Burden",
            [
                "Doctors and nurses spend a large share of their working day on paperwork. "
                "Speech recognition combined with language models can draft clinical notes "
                "from a recorded consultation, suggest billing codes and summarise a long "
                "patient history before an appointment. Early studies report that clinicians "
                "using such tools spend less time at the keyboard and feel less burned out, "
                "although every generated note must still be reviewed for accuracy.",
                "Hospitals also use optimisation algorithms to forecast admissions, schedule "
                "operating theatres and manage bed capacity. Better forecasting reduces "
                "waiting times and avoids both overcrowded wards and idle resources, which "
                "lowers cost without changing the care a patient receives.",
            ],
        ),
        (
            "6. Personalised Medicine and Remote Monitoring",
            [
                "People differ in their genes, lifestyle and response to drugs. Machine "
                "learning on genomic and clinical data helps identify which treatment is "
                "most likely to work for a particular patient, a field known as personalised "
                "or precision medicine. In cancer care, for example, the genetic profile of "
                "a tumour can guide the choice of therapy.",
                "Wearable devices and home sensors stream heart rate, blood glucose, sleep "
                "and activity data. Algorithms can detect an irregular heart rhythm or a "
                "dangerous trend in blood sugar and notify the patient or the care team. "
                "Remote monitoring lets people with chronic illnesses stay at home, extends "
                "care to rural communities and can prevent unnecessary hospital visits.",
            ],
        ),
        (
            "7. Summary of the Main Benefits",
            [
                "Taken together, the main benefits of artificial intelligence in healthcare "
                "are earlier and more accurate detection of disease, faster handling of "
                "urgent cases, support for clinicians' decisions, shorter and cheaper drug "
                "development, less administrative work, more personalised treatment, wider "
                "access to expertise in under-served regions and better use of scarce "
                "hospital resources. These gains are real but are not automatic. They depend "
                "on high-quality data, careful integration into clinical workflows and "
                "continuous evaluation.",
            ],
        ),
        (
            "8. Challenges and Risks",
            [
                "Data quality and bias are the most serious technical challenges. A model "
                "trained mostly on patients from one country, age group or ethnicity may "
                "perform worse for others. A well-known example is skin-lesion software "
                "trained mainly on light skin tones, which is less reliable on dark skin. "
                "Health records are also incomplete, inconsistent between hospitals and "
                "often contain recording errors, so cleaning and harmonising data is costly.",
                "Explainability is a further concern. Clinicians are reluctant to act on a "
                "recommendation that they cannot understand, and patients have a right to "
                "know how decisions about their care are reached. Techniques that highlight "
                "the image regions or record entries behind a prediction help, but they do "
                "not make complex models fully transparent.",
                "Regulation and accountability must also be settled. Software that "
                "influences diagnosis or treatment is regulated as a medical device in many "
                "countries and must show evidence of safety and effectiveness. When an "
                "error occurs, responsibility has to be shared clearly between the "
                "developer, the hospital and the clinician. Models must be monitored after "
                "release, because changes in equipment, treatment guidelines or patient "
                "mix can reduce accuracy over time.",
            ],
        ),
        (
            "9. Patient Privacy and Data Protection in Health AI",
            [
                "Health information is among the most sensitive personal data. Training and "
                "running medical models requires large amounts of it, which makes "
                "protection of patient privacy a central duty. Laws such as the United States "
                "Health Insurance Portability and Accountability Act and the European "
                "General Data Protection Regulation set strict rules for collecting, "
                "storing and sharing health data and require a lawful basis, often explicit "
                "consent, for its use.",
                "Practical safeguards include removing direct identifiers such as names and "
                "addresses before data is used for research, encrypting records in storage "
                "and in transit, restricting access to people who need it and keeping audit "
                "logs of every access. Privacy-preserving training methods, for example "
                "federated learning, allow hospitals to improve a shared model without "
                "sending raw patient records to a central server. Even de-identified records "
                "can sometimes be matched back to individuals by combining them with other "
                "datasets, so de-identification is a risk reduction rather than a guarantee.",
            ],
        ),
        (
            "10. Responsible Deployment",
            [
                "Responsible adoption starts with a clearly defined clinical problem and "
                "involves clinicians, patients and data specialists from the beginning. "
                "The model should be tested on local data before use, introduced first in a "
                "limited pilot, and always paired with human oversight so that a qualified "
                "professional can question or override its output. Teams should publish "
                "the intended use, the training data and known limitations of every system, "
                "train staff to use it appropriately and monitor outcomes after launch.",
                "Looking ahead, foundation models that combine text, images and laboratory "
                "values may support more general assistants, but the same principles apply: "
                "demonstrated benefit, fairness across patient groups, transparency and "
                "protection of privacy are the conditions for earning the trust on which "
                "healthcare ultimately depends.",
            ],
        ),
    ],
}

# ---------------------------------------------------------------------------
# Document 3: Data security and privacy
# ---------------------------------------------------------------------------
DOCUMENTS["data_security_and_privacy"] = {
    "title": "Data Security and Privacy in the Age of AI",
    "subtitle": "Principles, threats and safeguards for data-driven systems",
    "sections": [
        (
            "1. Security and Privacy: Two Related Goals",
            [
                "Data security and data privacy are closely connected but not identical. "
                "Security is about protecting information from unauthorised access, "
                "alteration or destruction. Privacy is about the appropriate and fair use of "
                "personal information, including what is collected, why, who may see it and "
                "how long it is kept. An organisation can be secure yet violate privacy, for "
                "instance by securely storing data that it had no right to collect.",
                "Security practice is often summarised by the confidentiality, integrity and "
                "availability triad. Confidentiality means only authorised people can read "
                "the data. Integrity means the data cannot be changed without detection. "
                "Availability means the data and the systems that process it can be reached "
                "when needed. Artificial intelligence systems add new ways to fail at each of "
                "these three goals, because they depend on very large datasets and on models "
                "that may themselves contain information about their training data.",
            ],
        ),
        (
            "2. The Main Challenges of Data Security and Privacy",
            [
                "The main challenges of data security and privacy can be grouped into a "
                "few themes. The first is scale: organisations now collect and copy data "
                "across cloud services, laptops, partners and analytics platforms, so it is "
                "hard to know where every sensitive record lives. The second is the "
                "sophistication of attackers, who use phishing, stolen credentials and "
                "software vulnerabilities to break in, and who increasingly target the "
                "supply chain of suppliers and open source components.",
                "The third challenge is the human factor. Most breaches involve a mistake "
                "such as a misconfigured storage bucket, a reused password or an employee "
                "sending a file to the wrong recipient. The fourth is the tension between "
                "usefulness and protection: richer data produces better models, yet every "
                "extra field increases the harm of a leak. The fifth is regulatory "
                "complexity, since companies that serve customers in many countries must "
                "satisfy several overlapping laws at the same time.",
            ],
        ),
        (
            "3. Threats Specific to Machine Learning Systems",
            [
                "Data poisoning occurs when an attacker inserts misleading examples into the "
                "training data so that the finished model behaves badly, for example by "
                "misclassifying a particular input. Systems that learn from user "
                "submissions or from data scraped from the internet are especially exposed.",
                "Adversarial examples are inputs that have been altered in a way that is "
                "almost invisible to a human but that causes the model to make a confident "
                "mistake, such as a stop sign sticker that fools a vision system. Model "
                "extraction attacks send many queries to a deployed model and use the "
                "answers to build a copy, stealing valuable intellectual property.",
                "Privacy attacks are a growing concern. In a membership inference attack the "
                "adversary tries to determine whether a particular person's record was used "
                "to train the model. In a model inversion attack the adversary reconstructs "
                "typical features of the training data from the model's outputs. Large "
                "language models can sometimes reproduce fragments of their training text "
                "word for word, which is dangerous if that text contained personal data. "
                "Applications built on language models face prompt injection, where hidden "
                "instructions in a document or web page trick the system into leaking "
                "information or taking unwanted actions.",
            ],
        ),
        (
            "4. Protecting Data Through Its Life Cycle",
            [
                "Data should be protected from collection to deletion. Encryption in "
                "transit, using protocols such as TLS, prevents eavesdropping while data "
                "moves across networks. Encryption at rest protects stored files and "
                "databases if a disk or backup is stolen. Encryption keys must themselves be "
                "managed carefully, kept separate from the data they protect, rotated "
                "regularly and held in a dedicated key management service.",
                "Access control limits who can reach what. The principle of least privilege "
                "gives every person and every program only the permissions needed for its "
                "job. Role-based access control, multi-factor authentication and the "
                "removal of accounts when people change roles shrink the damage a single "
                "stolen password can cause. Detailed audit logs record who accessed which "
                "records, which allows suspicious behaviour to be detected and investigated.",
                "Secure development and operations complete the picture: patching known "
                "vulnerabilities promptly, scanning dependencies, separating development "
                "from production data, taking tested backups and keeping a documented "
                "incident response plan so that a breach is contained quickly and affected "
                "people are notified on time.",
            ],
        ),
        (
            "5. Core Privacy Principles",
            [
                "Modern privacy law is built on a small set of principles. Data "
                "minimisation says an organisation should collect only what is necessary for "
                "a stated purpose. Purpose limitation says data gathered for one reason "
                "should not quietly be reused for an unrelated one. Transparency requires "
                "clear notices that explain what is done with personal information, and "
                "consent, where it is the legal basis, must be informed, specific and as easy "
                "to withdraw as to give.",
                "Storage limitation requires that data be deleted once it is no longer "
                "needed. Individuals are granted rights to access their data, to have errors "
                "corrected and, in many jurisdictions, to have their data erased. Privacy by "
                "design asks teams to build these protections into a system from the start "
                "instead of adding them after a problem has appeared.",
            ],
        ),
        (
            "6. Anonymisation, Pseudonymisation and Re-identification",
            [
                "Removing obvious identifiers such as names and phone numbers is called "
                "pseudonymisation when a separate key can restore the link, and "
                "anonymisation when the link is truly destroyed. True anonymisation is "
                "difficult. Researchers have repeatedly shown that combinations of ordinary "
                "attributes, such as postcode, birth date and gender, can single out a large "
                "share of the population, and that supposedly anonymous datasets can be "
                "re-identified by joining them with public records.",
                "Techniques such as generalisation, which replaces an exact age with an age "
                "band, suppression of rare values and k-anonymity, which requires that every "
                "record be indistinguishable from at least k minus one others, reduce the "
                "risk. They should be combined with legal and technical controls on who may "
                "receive the data, rather than treated as a complete solution.",
            ],
        ),
        (
            "7. Privacy-Preserving Machine Learning",
            [
                "Differential privacy offers a mathematical guarantee. Carefully calibrated "
                "random noise is added to the training procedure or to published statistics "
                "so that the result is almost the same whether or not any one individual's "
                "record is included. The strength of the guarantee is controlled by a "
                "privacy budget: a smaller budget gives stronger protection but lowers the "
                "accuracy of the model.",
                "Federated learning trains a shared model across many devices or "
                "organisations while keeping the raw data where it was created. Each "
                "participant computes an update locally and only the updates are combined. "
                "Because updates can still leak information, federated learning is often "
                "paired with secure aggregation and differential privacy. Other tools include "
                "homomorphic encryption, which allows computation on encrypted data, and "
                "synthetic data that mimics the statistics of real records without containing "
                "them. Each method brings a cost in computing power, complexity or accuracy.",
            ],
        ),
        (
            "8. Laws and Standards",
            [
                "The European General Data Protection Regulation applies to organisations "
                "that handle the personal data of people in the European Union, wherever the "
                "organisation is based, and allows large fines for serious violations. The "
                "California Consumer Privacy Act and similar state laws give residents "
                "rights to know, delete and opt out of the sale of their data. Sector rules "
                "such as the Health Insurance Portability and Accountability Act protect "
                "medical information, and payment card standards govern the handling of card "
                "numbers.",
                "Security management standards such as ISO 27001 and the NIST Cybersecurity "
                "Framework give organisations structured checklists of controls. Newer "
                "rules aimed specifically at artificial intelligence, including risk-based "
                "regulation in the European Union, add duties of documentation, human "
                "oversight and data governance for systems that affect people's rights.",
            ],
        ),
        (
            "9. Governance, Culture and Incident Response",
            [
                "Technology alone does not protect data. Effective organisations assign "
                "clear ownership, keep an inventory of the personal data they hold, perform "
                "privacy impact assessments before launching new projects and vet the "
                "security of every supplier. Regular training teaches staff to recognise "
                "phishing and to handle sensitive material correctly. A growing risk is "
                "shadow AI, where employees paste confidential text into public chatbots "
                "that the company has not approved.",
                "When an incident does occur, speed and honesty matter. A rehearsed plan "
                "defines who investigates, how the affected systems are isolated, when "
                "regulators and individuals must be told and how lessons are fed back into "
                "improved controls.",
            ],
        ),
        (
            "10. A Practical Checklist for Teams Building AI Systems",
            [
                "Teams that build data-driven products can use a short checklist. Collect "
                "only the data that the task truly needs. Document the source and the legal "
                "basis of every dataset. Encrypt data at rest and in transit and restrict "
                "access using least privilege. Remove or pseudonymise identifiers before "
                "training and test how easily records could be re-identified. Consider "
                "differential privacy or federated learning for sensitive domains. "
                "Validate and monitor training data for poisoning, and test deployed models "
                "against extraction and inference attacks. Log access, plan for incidents and "
                "delete data when its purpose has ended. Reviewing this list at each stage of "
                "a project keeps security and privacy part of everyday engineering.",
            ],
        ),
    ],
}
