 🩺 AI-Powered Healthcare Assistant

An intelligent and interactive Python-based healthcare assistant that predicts possible diseases based on symptoms, suggests treatments, shows health-related visualizations, and supports multilingual & voice responses. It also simulates IoT health data like heart rate monitoring.

---

 🚀 Features

- ✅ Symptom-based disease prediction using Machine Learning
- 🗣 Voice response in English using pyttsx3
- 🌐 Multilingual translation of results using Google Translate
- 📊 Health visualizations:
  - Fever temperature trend graph
  - Prediction confidence chart
  - IoT-simulated heart rate monitor
  - Diet recommendation chart
- 🥗 Suggests treatments and personalized diet plans
- 🔁 Collects user feedback for improvement
- 🧠 Trained using a Naive Bayes classifier with TF-IDF features

---

 🛠 Technology Used

| Category            | Tools / Libraries                         |
|---------------------|-------------------------------------------|
| Programming Language| Python 3                                  |
| ML Model            | Naive Bayes (MultinomialNB from sklearn)  |
| Data Processing     | scikit-learn, TF-IDF                      |
| Visualization       | Matplotlib                               |
| Voice Output        | pyttsx3                                   |
| Translation         | googletrans                               |
| Simulated IoT       | Random module, Matplotlib                 |

---

 🔄 How It Works

1. *User Input*: The user enters their name, age, language, and symptoms.
2. *Symptom Processing*: Valid symptoms are matched against known mappings.
3. *Disease Prediction*: A trained Naive Bayes model predicts the disease based on symptoms.
4. *Voice & Translation*: The output is spoken aloud and translated to the user's.
5. *Visual Insights*: Displays graphs including:
   - Fever trend (if symptom includes fever)
   - Confidence of prediction
   - Simulated heart rate (IoT)
   - Diet recommendation
6. *Feedback*: User can give feedback for further improvement.

---

 📊 Data Collection

- The dataset is synthetically generated based on predefined symptom-disease mappings.
- Each disease is mapped with 2-3 relevant symptoms.
- Data is expanded by shuffling and recombining symptoms multiple times to simulate real-world variation.
- You can later improve accuracy by integrating a larger dataset from medical sources like:
  - [Kaggle:Disease and Symptoms dataset](https://www.kaggle.com)
  - Government Health Portals or WHO datasets

---

 📌 How to Run

1. Install dependencies:
   ```bash
   pip install matplotlib
   scikit-learn
   googletrans==4.0.0-rc1
   pyttsx3
   pip install --upgrade httpx
   pip install deep-translator
