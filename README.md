# farzana-zaki.github.io
# Data Science Portfolio - Farzana Rahmat Zaki
This Portfolio is a compilation of all the Data Science and Data Analysis projects I have done for academic, professional, and self-learning purposes. It also contains my Achievements, skills, and certificates. It is updated regularly.

- **Email**: farzana.rahmat.zaki@gmail.com
- **LinkedIn**: https://www.linkedin.com/in/farzana-zaki-9bba43125/


# Project-1: Gen-AI-applications-with-Langchain

# Project Name: Langchain Data Science Assistant (Github link: https://github.com/farzana-zaki/Gen-AI-applications-with-Langchain)

## Business Context:
Perhaps the largest and most prominent application of Large Language Models has been their ability to converse with human users in coherent Natural Language and function as helpful assistants for organizational productivity. As we know by now, a lot of work goes into creating and providing such Assistant-style LLMs to us as a service - the hugely compute-expensive Supervised Pre-training stage that's responsible for the vast majority of the LLM's world knowledge, the Instruction-style Supervised Fine-tuning stage on curated dialog datasets that adapts the LLM's behavior to that of a high-quality assistant, and the RLHF stage that teaches the LLM to provide responses that are aligned with human values. The archetypal assistant-style Generative AI solution today is probably an LLM-powered chatbot that can query our databases, analyze our data, and even fetch our answers to contextual questions from documents or the web.

In this Final Project, I have applied a Data Analysis / Data Science twist to this functionality. I have used the LangChain framework on top of an LLM to create a Data Science Assistant that will automate the following tasks for us by converting the Natural Language requests we provide into the code required to execute the task:

(a) Data Analysis, (b) Data Science & Machine Learning Modeling, (c) Relational Database Querying, (d) Retrieval-Augmented Generation for Contextual Answers

There are obvious reasons why such a project could be highly significant to business/research. By automating the code tasks common to various operations in Data Analysis, Data Science, or Database Querying, such a solution has the potential to save significant time, improve code accuracy, and also open up Data Analysis and Data Science to a more general audience that may not be familiar with the code involved.

From a broader perspective, the newfound ability of the best large language models (beyond 1 billion parameters) to understand the true meaning of Natural Language commands and coherently respond to, acknowledge, and fulfill them has enabled us to usher in the next wave in the long list of automation that Computer Science has seen in its history. Translating commands into programming code, once solely the province of human developers, is now increasingly being automated by Generative AI, and we'll see many examples of the same in this project.

## Project Objective:
In this project, a LangChain agent has been utilized as a Data Science Assistant and a Data Retriever Assistant, with the capacity to perform a range of tasks, including data analysis, machine learning modeling, relational database querying, and RAG generation for contextual questions from documents or the web. The agent keeps generating commands until it obtains the response the user wants, iterating until it satisfies their question or reaches the set limit.

## Recommendations:
• This project can significantly impact businesses and research sectors, as it automates repetitive tasks and frees up more time for strategic work in the realms of Data Analysis, Data Science, and Database Querying.

• For example, LangChain agents can automatically summarize customer emails, generate reports, generate or edit codes, or schedule meetings according to user preferences. This type of assistant can save significant time and improve code accuracy without necessitating extensive coding expertise.

• Additionally, it can expand the accessibility of Data Analysis and Data Science to individuals who may not be fully familiar with the involved coding procedures. LangChain agents are designed to be intuitive and user-friendly.

• LangChain agents possess the capability to handle dynamic situations and adapt their actions based on new information or user feedback, enhancing the user experience and expediting task completion in a more intuitive, efficient, and collaborative manner.

Through the utilization of language models via LangChain agents, we are on the edge of a new era of automation, efficiency, and collaboration. This technology is a powerful asset that has the potential to revolutionize the standard practices of various industries.



# Project-2: Generative-AI-for-NLP

# Project Name: Generative AI-powered Support Ticket Categorization (Github link: https://github.com/farzana-zaki/Generative-AI-powered-Support-Ticket-Categorization)

## Business Context:
In today's dynamic business landscape, organizations are increasingly recognizing the pivotal role customer feedback plays in shaping the trajectory of their products and services. The ability to swiftly and effectively respond to customer input not only fosters enhanced customer experiences but also serves as a catalyst for growth, prolonged customer engagement, and the nurturing of lifetime value relationships.

As a dedicated Product Manager or Analyst, staying attuned to our customers' voices is not just a best practice; it's a strategic imperative.
While the organization may be inundated with customer-generated feedback and support tickets, our role entails more than processing these inputs. To make our efforts in managing customer experience and expectations truly impactful, we need a structured approach—a method that allows us to discern the most pressing issues, set priorities, and allocate resources judiciously.

One of the most effective strategies we can employ as an organization is to harness the power of automated Support Ticket Categorization, done today using large language models and Generative AI.

## Project Objective:
Develop a Generative AI application using a Large Language Model to automate the classification and processing of support tickets. The application will aim to predict ticket categories, assign priority, suggest estimated resolution times, generate responses based on sentiment analysis, and store the results in a structured data frame.

## Observations and insights:
The aim of this project is to create a Generative AI application that can automate the processing and classification of support tickets using a Large Language Model. To achieve this, I have utilized Llama 2, a pre-trained and fine-tuned generative text model ranging from 7 billion to 70 billion parameters. Specifically, I have used the 13B fine-tuned model (llama-2-13b-chat.Q6_K.gguf, which is a 6-bit quantized version model) that is optimized for dialogue use cases, and I have converted it to the Hugging Face Transformers format.

I merged the support ticket text from the input CSV file with a system message to create a prompt for the technical assistant. The system message contains detailed instructions and guidelines to help the technical assistant classify the support ticket text, generate tags, assign priority, suggest the expected time of arrival (ETA), and create a sentiment-based response in JSON format. The system message also includes a one-shot example to guide the technical assistant.

Next, a response text from the LLaMA model was generated using the lcpp_llm instance with the following parameters: max_tokens = 256, temperature = 0, top_p = 0.95, repeat_penalty = 1.2, top_k = 50, and echo = False.

The generated LLaMA response includes some additional text before the JSON format response for each ticket. Therefore, text trimming was performed from the generated LLaMA response to keep only the JSON response.

Next, I parsed the JSON data, extracted key-value pairs, and normalized and concatenated them (setting axis = 1) in the 'data' DataFrame to add the JSON response as 5 new columns (category, tags, priority, suggested ETA, and generated 1st response) in the 'data' DataFrame for better visualization.

## Business Recommendations:
The objective of the application is to simplify the management of support tickets by incorporating advanced tools that automate the classification and processing of incoming tickets. The Llama 2 application model uses generative capabilities to process support tickets efficiently and effectively, which reduces response time and minimizes error rates. Overall, this project is a significant step forward in supporting ticket management, leveraging state-of-the-art AI technologies to enhance a crucial business function.

Fine-tuning the prompt with a few shot examples can be very helpful in enhancing the performance of the LLM models and adapting their outputs to specific business requirements. Additionally, tweaking the prompt's parameters, such as Temperature, Top K, Frequency Penalty, etc., is another way to modify the LLM's outputs according to the specific business needs.


# Project-3: Machine Learning Project (IBM Data Science Capstone)

# Project Name: Winning Space Race with Data Science (Github link: https://github.com/farzana-zaki/Winning-Space-Race-with-Data-Science)

## Project background and context:
SpaceX advertises Falcon 9 rocket launches on its website, which cost 62 million dollars; other providers cost upward of 165 million dollars each. Much of the savings is because SpaceX can reuse the first stage.
Therefore, if we can determine if the first stage will land, we can determine the cost of a launch. This information can be used if an alternate company wants to bid against SpaceX for a rocket launch. The goal of
this project is to develop a machine learning pipeline which will predict if the rocket will pass the first stage successfully.

## Problem statements:
❑ Major factors/features to determine the landing of the first stage of the rocket successfully.
❑ The operating conditions to ensure a successful landing of the rocket launch.

## Executive Summary:
• Data collection methodology: Data was collected using SpaceX API and web scraping from Wikipedia.
• Perform data wrangling: One-hot encoding for categorical features of the machine learning, data cleaning of null values and irrelevant columns.
• Perform exploratory data analysis (EDA) using visualization and SQL: EDA using Python libraries (Matplotlib, Seaborn) and SQL, feature engineering.
• Perform interactive visual analytics using Folium andPlotlyDash: Applied Folium to view previously observed correlations interactively.
• Perform predictive analysis using classification models: Classification models (Logistic regression, Support vector machine, K-nearest neighbor, and Decision Tree) have been built and evaluated using GridSearch 
  for the best classifier.

  ## Conclusion: 
• KSC LC-39A had the most successful launches of any site.
• The larger the flight amount at a launch site, the greater the success rate at a launch site.
• Launch success rate started to increase in 2013 till 2020.
• Most successful launch outcomes are observed in the 2000 – 5300 kg payload range.
• The FT Booster version had the most successful launch outcomes, with a success rate of around 71% in the 2000 – 5300 kg payload range and around 64% success rate in the 500 – 9500 kg payload range.
• Orbits ES-L1, GEO, HEO, SSO, and VLEO had the most success rate.
• Both Logistic Regression and Decision tree classifiers showed the optimal machine learning algorithms for the provided dataset, with a test accuracy of 94.44%.



# Project-4: Deep Learning Project (MIT Applied Data Science Capstone)

# Project Name: Facial emotion detection using ANN, CNN, and Transfer Learning (Github link: https://github.com/farzana-zaki/Facial-emotion-detection)

## Business Context:
Facial emotions and their analysis are essential for detecting and understanding human behavior, personality, mental state, etc. Most people can recognize facial emotions quickly regardless of gender, race, and 
nationality. However, extracting features from face-to-face is a challenging, complicated, and sensitive task for computer vision techniques, such as deep learning, image processing, and machine learning to
perform automated facial emotion recognition and classification. Some key challenges in extracting features from the facial image dataset include a variation of head pose, resolution and size of the image, background, and presence of other objects (hand, ornaments, eyeglasses, etc).In recent years, deep learning has become an efficient approach with the implementation of various architectures that allow the automatic extraction of features and classification using convolutional neural network (CNN), transfer learning, and recurrent neural network (RNN). This project aims to build a CNN model for accurate facial emotion detection.

## Objective: 
The purpose of this project is to build a CNN model for accurate facial emotion detection. The proposed CNN model performs multi-class classification on images of facial emotions to classify the expressions
according to the associated emotion.

## Solution summary:
The proposed CNN model performs multi-class classification on images of facial emotions, classifying the expressions according to the associated emotion.

In this facial emotion detection project, various CNN models (simple, transfer learning, and complex) are employed for training, validation, and testing to observe their accuracies in detecting those emotions.
In total, 17 different configuration CNN models were applied and evaluated. Simple and transfer learning CNN models are overfitting and have low F1 scores.

For building the proposed CNN model, the hyperparameter tuning using a random search from the Keras tuner was applied to select the building blocks of the complex CNN models. Adam with three learning rates, 0.1,0.01, and 0.001, are used as an optimizer. A layer with five convolutional blocks for feature selection and three dense layers for classification is used for the complex CNN models with batch sizes of 16,32 and 64. Out of three complex CNN models, model 6a(CNN model with five convolutional blocks for feature selection and three dense layers for the classification, with batch size of 32, learning rate of 0.001, and Adam optimizer) shows the best performance. Model 6a is selected as the final proposed CNN model for face emotion detection.

The final proposed model solved the overfitting problem and is well generalized and optimized with training, validation, and overall test accuracies of 72.23%,69.10%, and 74%, respectively. This model has achieved an average F1 score of 0.74. Batch normalization and dropout are used to solve the overfitting problem.

However, the model has poor performance for detecting class-1 (neutral) and class-2 (sad) with F1 scores of 0.70 and 0.56, respectively.

## Key recommendations and future implementation:
(1) The training dataset is slightly imbalanced. However, the validation dataset is pretty much imbalanced. Total number of images of the four classes (Happy:1825, Neutral:1216, Sad:1139, Surprises:797) in the validation dataset. As we can see, 'surprise' and data have less frequency (0.16) compared to the other three emotions. So, the validation dataset is imbalanced due to the 'surprise' dataset. We could employ an
oversampling technique for the 'surprise' dataset to make a balanced dataset and then again train the model and compare the performances.

(2)Also, there are some poor-quality images in the training dataset. For example, some images contain watermarked text, and some training images do not have any facial expressions (rather have question marks or cross signs instead of any image). Images of neutral and sad faces are pretty much confusing. Therefore, CNN algorithms have faced the difficulty of correctly detecting them properly. Therefore, for all the classifiers, F1 scores of neutral and sad emotions are not satisfactory.

(3)The dataset is pretty small. Data augmentation can generate a large volume of training datasets by using the transformations of the face images, such as flip, shift, scaling, and rotation.

(4)Several experiments need to be carried out with mode convolutional layers (such as 6 or more layers) to verify the effectiveness of the augmented dataset and the performance of these approached CNN models in
comparison with some of the frequently used face recognition methods.

(5)The proposed CNN model was implemented using GPU. Using deeper convolutional layers with millions to trillions of training datasets may increase the implementation cost.

(6)Several other transfer learning models can be applied to improve the performance of facial recognition.

(7)It is recommended that stakeholders consider these variables in building improved long-term facial emotion detection models, as well as include the full range of environmental implications of various data
sources, usage of color images instead of grayscale images, and usage of more training data in developing future facial emotion detection.



# Project-5: Deep Learning Project (MIT Applied Data Science)

# Project Name: SVHN Digit Recognition using ANN and CNN (Github link: https://github.com/farzana-zaki/SVHN-Digit-Recognition-using-ANN-and-CNN)

## Business Context:
One of the most interesting tasks in deep learning is to recognize objects in natural scenes. The ability to process visual information using machine learning algorithms can be very useful, as demonstrated in various applications. The SVHN dataset contains over 600,000 labeled digits cropped from street-level photos. It is one of the most popular image recognition datasets. It has been used in neural networks created by Google to improve the map quality by automatically transcribing the address numbers from a patch of pixels. The transcribed number with a known street address helps pinpoint the location of the building it represents.

## Objective:
The objective is to predict the number depicted inside the image using Artificial or Fully Connected Feed Forward Neural Networks and Convolutional Neural Networks. I used various ANN and CNN models and, finally
selected the one that provided me with the best performance. 

## Observations:
The CNN model gives about 91% accuracy on the test data. However, there are a few misclassifications between all classes. The average f1-score is much improved for the CNN model with 0.91. Previously, for ANN,
f1-score was 0.77. Digits '0' and '4' have the highest f1-score with 0.93, whereas digit '3' has the lowest f1-score of 0.88 (for ANN, it was 0.65, around 35% improvement of f1-score for the digit '3').
Again, the precision of digit '3' is the lowest (only 0.86) among all other digits for CNN but around 32% improvement of precision for digit '3' in CNN is observed in the model when compared to the ANN precision of digit '3' (precision = 0.65). A large number of images of digits '1' and '2' were predicted as digits '7', digit '3' was identified as '5' and vice versa, digit '4' was misclassified as digit '1', digit '6' as digit '5'and digit '8' as digit '3', '6' and '9'. Overall, CNN is better than ANN as a digit recognizer in this project.


# Project-6: Exploratory Data Analysis Project (MIT Applied Data Science)

# Project Name: Analyzing different aspects of Diabetes in the Pima Indian tribe  (Github link: https://github.com/farzana-zaki/EDA-Prima-Indian-Diabetes)

## Background:
Diabetes is one of the most frequent diseases worldwide, and the number of diabetic patients has grown over the years. The main cause of diabetes remains unknown, yet scientists believe that both genetic factors and environmental lifestyle play a major role in diabetes. A few years ago, research was done on a tribe in America called the Pima tribe (also known as the Pima Indians). In this tribe, it was found that the ladies were prone to diabetes very early. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients were females at least 21 years old of Pima Indian heritage. 

## Objective:
In this project, I have analyzed different aspects of Diabetes in the Pima Indian tribe by doing Exploratory Data Analysis (univariate, bivariate).

## Summary of observations:
(1) In this EDA, I have used Python packages such as Numpy, Pandas, Matplotlib, and Seaborn. Numpy, also known as 'Numerical Python', is a Python library used for arrays. Pandas, also known as 'Panel data,', is a software library for the Python programming language. Pandas are used for data cleaning, data transformation, data manipulation, data visualization, and data analysis. Matplotlib is a data visualization library in python used for plotting graphs. Seaborn is a data visualization library in Python based on Matplotlib. Seabon is useful for visualizing univariate and bivariate data.

(2) There are no missing values in the Pima dataset.

(3) Summary statistics of the data represent the overall quantitative summary statistics for all variables of the dataset. It computes the total number of data points, mean, standard deviation, minimum and maximum values, 1st,2nd(median), and 3rd quartiles for each column of the dataset. For example, if we consider the BMI column from the dataset, there are 768 data points with a mean BMI of 32.45 and a standard deviation of ~6.88. The BMI range varies from 18.2 to 67.1 with 1st, 2nd(median), and 3rd percentiles of 27.5, 32, and 36.6, respectively.

(4) Pairplot generates a matrix of axes representing the relationship for each column pair in a dataset. In the diagonal axes, it also creates the univariate distribution of each variable. Here, on the diagonal axes, we can observe the univariate distribution of each glucose, skin thickness, and Diabetes pedigree function based on 2 outcomes (0 represents non-diabetic and 1 represents diabetic). Moreover, six other plots are the bivariate distribution between glucose-skin thickness, glucose-diabetes pedigree function, and skin thickness-diabetes pedigree function, repeated twice based on two different toned outcomes (0 represents non-diabetic and 1 represents diabetic).

(5) Most insulin levels are around 100-200, and glucose levels range from 80-150. Also, this scatterplot does not observe a significant correlation between insulin and glucose.

(6) Based on the 1st histogram, most of the diabetic women groups in the 20-29years age group, and the least diabetic women age group is 60-69years. Based on the 2nd histogram, most of the non-diabetic women groups are in the 20-29 age group, and the least non-diabetic women age group is 70-79 years. For both cases, women aged 20-29 are prominent. Also, both histograms are right-skewed. However, non-diabetic data contains a heavy tail.

(7) From the heatmap, the color bar is helpful in determining the most correlated attributes easily. The lighter the color, the higher the correlation in this heatmap. Most cells have relatively low correlation values (except diagonal 1; those are autocorrelation values. So it will always be 1 for all cases). However, higher correlations are observed between pregnancy age (r = 0.54) and BMI-sickness(r=0.53).







