# Big Brother Debunker
## Fake News Detection
<img src="https://github.com/user-attachments/assets/6936d7c2-76e9-4ed8-9322-8a307df2fc8d" width="400"/>

# 1. Problem Statement
Misinformation in the media, exemplified by events like the Arab Spring, can influence public opinion and incite social unrest. Traditional verification methods, reliant on human effort, struggle with the vast volume of online content.

This project aims to create a scalable tool that detects and classifies fake news, assisting users and tech companies in identifying reliable information and reducing the spread of misinformation. We proposes an automated solution to detect fake news by analysing textual features such as words, phrases, sources, and titles using machine learning. Utilizing a labelled dataset from Kaggle, we applied supervised machine learning algorithms and feature selection techniques to develop a predictive model. The model's accuracy is determined through testing on unseen data.

# 2. Methodology and Data Preprocessing
## 2.1. Methodology
We adhere to a structured approach consisting of data pre-processing, feature engineering, classification, and analysis. The detailed steps are as follows:
![process](https://github.com/user-attachments/assets/d44fd337-1029-4b7b-9fc6-c5893aee7a3b)

## 2.2. Data Preprocessing
- We begin by checking for missing values (NAs) across all columns and confirm that the label column contains no NAs. Since our goal is to extract additional attributes from other columns, there is no need to remove any columns at this initial stage.
    - From the label distribution, fake news and non-fake news are evenly distributed. We got balanced dataset and don't need to oversample.
      
**Shape**
- Rows: 20800
- Columns: 5
    - id, title, author, text, label

**Missing values**

- For the titles and text, we pre-process by removing non-English words, special characters, HTML tags, and stopwords. Additionally, we standardize text casing, perform lemmatization to reduce words to their base forms, strip extra spaces, and filter out common but uninformative words to enhance the clarity and effectiveness of our data for classification tasks.
- Split news source from the title. Source refers to the organization where the news is coming from. For example, The New York Times. However, only part of the title have source information. We believe containing source information or not is caused by crawling, which is irrelevant to whether one news is fake.
- For example, id=80:
  
    - **Before**:
      
        Louisiana, Simone Biles, U.S. Presidential Race: Your Tuesday Evening Briefing - The New York Times
      
    - **After**:
      
        Louisiana, Simone Biles, U.S. Presidential Race: Your Tuesday Evening Briefing

- Stemming: Stemming is performed to reduce words to their base form, thereby decreasing the number of word variations. This helps machine learning models better understand and process language data because different forms of the same root word (such as "run", "running", "runner") are semantically similar. With stemming, the model can treat these variations as the same word, reducing the complexity of the feature space and improving processing efficiency and model performance.
  
- Explore the word count:
  
    -  Word Cloud for Titles: This image highlights the most prominent words used in titles or headings. Key terms include "Trump," "Hillary," "Clinton," and "Obama," indicating a political context. Other significant words such as "America," "President," "Russia," and "Media" suggest topics related to international relations, leadership, and news coverage. The size of the words "Trump" and "Hillary" suggests they are among the most frequently mentioned, pointing to a dataset possibly from the time around the 2016 U.S. Presidential election.
    
    - Word Cloud for Stemmed Texts: This cloud shows stemmed versions of words (base forms), which aids in grouping similar terms together, enhancing the text analysis by consolidating different forms of a word into a single representation. Dominant terms here include "Trump," "state," "presid" (likely stemming from "president"), and "peopl" (stemmed from "people"). This visualization emphasizes themes similar to the first cloud but from a broader textual analysis, potentially from article bodies or extensive texts discussing similar political themes.

# 3. Feature Engineering
