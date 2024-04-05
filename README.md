# farzana-zaki.github.io
# Data Science Portfolio - Farzana Rahmat Zaki
This Portfolio is a compilation of all the Data Science and Data Analysis projects I have done for academic, professional, and self-learning purposes. It also contains my Achievements, skills, and certificates. It is updated regularly.

- **Email**: farzana.rahmat.zaki@gmail.com
- **LinkedIn**: https://www.linkedin.com/in/farzana-zaki-9bba43125/


# Project-1: Gen-AI-applications-with-Langchain

# Project Name: Langchain Data Science Assistant (Github link: https://github.com/farzana-zaki/Gen-AI-applications-with-Langchain)

## Business Context
Perhaps the largest and most prominent application of Large Language Models has been their ability to converse with human users in coherent Natural Language and function as helpful assistants for organizational productivity. As we know by now, a lot of work goes into creating and providing such Assistant-style LLMs to us as a service - the hugely compute-expensive Supervised Pre-training stage that's responsible for the vast majority of the LLM's world knowledge, the Instruction-style Supervised Fine-tuning stage on curated dialog datasets that adapts the LLM's behavior to that of a high-quality assistant, and the RLHF stage that teaches the LLM to provide responses that are aligned with human values. The archetypal assistant-style Generative AI solution today is probably an LLM-powered chatbot that can query your databases, analyze our data and even fetch our answers to contextual questions from documents or the web.

In this Final Project, I have applied a Data Analysis / Data Science twist to this functionality. I have used the LangChain framework on top of an LLM to create a Data Science Assistant that will automate the following tasks for us by converting the Natural Language requests we provide into the code required to execute the task:

Data Analysis

Data Science & Machine Learning Modeling

Relational Database Querying

Retrieval-Augmented Generation for Contextual Answers

There are obvious reasons why such a project could be highly significant to business/research. By automating the code tasks common to various operations in Data Analysis, Data Science, or Database Querying, such a solution has the potential to save significant time, improve code accuracy, and also open up Data Analysis and Data Science to a more general audience that may not be familiar with the code involved.

From a broader perspective, the newfound ability of the best large language models (beyond 1 billion parameters) to understand the true meaning of Natural Language commands and coherently respond to, acknowledge, and fulfill them has enabled us to usher in the next wave in the long list of automation that Computer Science has seen in its history. Translating commands into programming code, once solely the province of human developers, is now increasingly being automated by Generative AI, and we'll see many examples of the same in this project.

## Project Objective:

In this project, a LangChain agent has been utilized as a Data Science Assistant and a Data Retriever Assistant, with the capacity to perform a range of tasks, including data analysis, machine learning modeling, relational database querying, and RAG generation for contextual questions from documents or the web. The agent keeps generating commands until it obtains the response the user wants, iterating until it satisfies their question or reaches the set limit.

## Recommendations:

• This project can significantly impact businesses and research sectors, as it automates repetitive tasks and frees up more time for strategic work in the realms of Data Analysis, Data Science, and Database Querying.

• For example, LangChain agents can automatically summarize customer emails, generate reports, generate, or edit codes, or schedule meetings according to user preferences. This type of assistant can save significant time and improve code accuracy without necessitating extensive coding expertise.

• Additionally, it can expand the accessibility of Data Analysis and Data Science to individuals who may not be fully familiar with the involved coding procedures. LangChain agents are designed to be intuitive and user-friendly.

• LangChain agents possess the capability to handle dynamic situations and adapt their actions based on new information or user feedback, enhancing the user experience and expediting task completion in a more intuitive, efficient, and collaborative manner.

Through the utilization of language models via LangChain agents, we are on the edge of a new era of automation, efficiency, and collaboration. This technology is a powerful asset that has the potential to revolutionize the standard practices of various industries.



# Project-2: Generative-AI-for-NLP

# Project Name: Generative AI-powered Support Ticket Categorization (Github link: [https://github.com/farzana-zaki/Gen-AI-applications-with-Langchain](https://github.com/farzana-zaki/Generative-AI-powered-Support-Ticket-Categorization))

# Business Context
In today's dynamic business landscape, organizations are increasingly recognizing the pivotal role customer feedback plays in shaping the trajectory of their products and services. The ability to swiftly and effectively respond to customer input not only fosters enhanced customer experiences but also serves as a catalyst for growth, prolonged customer engagement, and the nurturing of lifetime value relationships.

As a dedicated Product Manager or Analyst, staying attuned to our customers' voices is not just a best practice; it's a strategic imperative.
While the organization may be inundated with customer-generated feedback and support tickets, our role entails more than processing these inputs. To make our efforts in managing customer experience and expectations truly impactful, we need a structured approach—a method that allows us to discern the most pressing issues, set priorities, and allocate resources judiciously.

One of the most effective strategies we can employ as an organization is to harness the power of automated Support Ticket Categorization, done today using large language models and Generative AI.

# Project Objective
Develop a Generative AI application using a Large Language Model to automate the classification and processing of support tickets. The application will aim to predict ticket categories, assign priority, suggest estimated resolution times, generate responses based on sentiment analysis, and store the results in a structured data frame.

# Observations and insights:
The aim of this project is to create a Generative AI application that can automate the processing and classification of support tickets using a Large Language Model. To achieve this, I have utilized Llama 2, a pre-trained and fine-tuned generative text model ranging from 7 billion to 70 billion parameters. Specifically, I have used the 13B fine-tuned model (llama-2-13b-chat.Q6_K.gguf, which is a 6-bit quantized version model) that is optimized for dialogue use cases, and I have converted it to the Hugging Face Transformers format.

I merged the support ticket text from the input CSV file with a system message to create a prompt for the technical assistant. The system message contains detailed instructions and guidelines to help the technical assistant classify the support ticket text, generate tags, assign priority, suggest the expected time of arrival (ETA), and create a sentiment-based response in JSON format. The system message also includes a one-shot example to guide the technical assistant.

Next, a response text from the LLaMA model was generated using the lcpp_llm instance with the following parameters: max_tokens = 256, temperature = 0, top_p = 0.95, repeat_penalty = 1.2, top_k = 50, and echo = False.

The generated LLaMA response includes some additional text before the JSON format response for each ticket. Therefore, text trimming was performed from the generated LLaMA response to keep only the JSON response.

Next, I parsed the JSON data, extracted key-value pairs, and normalized and concatenated them (setting axis = 1) in the 'data' DataFrame to add the JSON response as 5 new columns (category, tags, priority, suggested ETA, and generated 1st response) in the 'data' DataFrame for better visualization.

# Business Recommendations:
The objective of the application is to simplify the management of support tickets by incorporating advanced tools that automate the classification and processing of incoming tickets. The Llama 2 application model uses generative capabilities to process support tickets efficiently and effectively, which reduces response time and minimizes error rates. Overall, this project is a significant step forward in supporting ticket management, leveraging state-of-the-art AI technologies to enhance a crucial business function.

Fine-tuning the prompt with a few shot examples can be very helpful in enhancing the performance of the LLM models and adapting their outputs to specific business requirements. Additionally, tweaking the prompt's parameters, such as Temperature, Top K, Frequency Penalty, etc., is another way to modify the LLM's outputs according to the specific business needs.



