# Assessment Spec

## Learning Outcomes

This assessment relates to the learning outcomes of the course which are:

- CLO 1: Apply data science to analyse social media and social networks.
- CLO 2: Analyse social networks by finding communities, identifying important nodes and influence propagation.
- CLO 3: Analyse social media by applying Natural Language Processing (NLP) techniques to detect sentiment and topics.
- CLO 5: Synthesise and present insights from the social media and network analysis performed.

## Assignment Details

The goals of the master programs and of RMIT graduates in general are the ability to problem solve, to work in teams and to be able to communicate. This assignment aims to contribute towards your development in these crucial areas. It is essentially a cross between a hackathon and a final year project, where you work on a problem to solve or answer in a team, and in a length of time that is between a hackathon and a final year project.

The assignment can be broken up into a few parts, corresponding to how you could approach it and also corresponding with the marking rubric. The parts are as follows:

- Team formation
- Problem/Question construction
- Team management & Mentoring
- Perform the analysis/Solve the problem
- Communicating the results or analysis

## Problem or Question Construction

You will also need to select the problem, analysis aim, or research question your team will work on. Your project may be problem-solving or research-oriented. Exploratory analysis is allowed, but it must still have a clear guiding aim and explain what would count as useful or convincing findings.

Each team must submit a topic proposal by Friday 8 May 2026, 11:59 pm AEST. The teaching team will provide the detailed topic proposal and consultation process on Canvas.

Your topic proposal and final report should make clear:

- What problem, question, or exploratory aim the team is investigating.
- Why it is worth investigating using social media and network analysis.
- What data source or type of data the team expects to use.
- What network will be constructed or analysed, including what the nodes and edges represent.
- What success criteria will be used to judge whether the analysis has answered the question, solved the problem, or produced useful exploratory insights.

As a guide about the scope of problems that are doable, consider the following:

- Beyond statistics, who is the best and most influential cricketer?
- Who is the centre of the Marvel/Nintendo universe? Can we promote Browser to be the most central Nintendo head honcho?
- Github & communities: all things about community analysis on Github.
- Donald vs Kim: who has the bigger ego?
- Social Media Neighbourhood Watch: alerts for where, when and what crimes are been committed in Melbourne?
- What is trending in the area of Climate change on social media?
- What are the characteristics of successful and popular designs on Canva?

If you look at the problems, some are more analysis/research in nature, others are about solving a problem. This is not an exhaustive list. There is no bias in assessment towards the different types, as we know each of you has different backgrounds, knowledge and interest, so would like to leave it open to you.

## Assignment Constraints

There are some constraints that your assignment should adhere to:

- Must include at least one source of social media.
- Must include a graph or network and its analysis/processing to solve a problem.
- [COSC 2671 Postgraduate Students Only] Must include network structure & influence analysis. For instance, you could consider: How do different network measures such as centrality, clustering and community structure reveal the roles of users in a social network, and how can these insights be used to identify influential nodes or communities?
- Must include some element of analysis using social media and networks, and something we learnt in class. It cannot be all machine learning, for example.
- Your analysis or solution is to be implemented primarily in Python. R is not permitted, including for visualisation submitted for assessment. Other languages may be used only for optional supporting components, such as a prototype front end.
- You can leverage packages such as NetworkX, etc., but it should not be copied from an existing solution.

## Dataset Requirements

Your dataset must be suitable for both graph/network analysis and NLP or text-based analysis. This means the data should contain enough relational information to construct or analyse a network, and enough textual content to support meaningful NLP, text, topic, sentiment, or similar social media analysis.

Acceptable data sources include social media platforms, online social platforms, public forums, online communities, public repositories, public datasets, or APIs, provided the data is appropriate for the project question and complies with relevant platform terms and legal or ethical restrictions. Examples may include Reddit, X/Twitter, YouTube comments, GitHub, public forums, or public benchmark datasets.

You may collect data yourself or use an existing public dataset. In either case, the report must clearly state:

- Where the data came from.
- How it was collected or obtained.
- The collection date or time period covered, where relevant.
- The main fields or variables used in the analysis.
- How the data supports the network analysis component.
- How the data supports the NLP or text analysis component.
- Any important limitations, missing data, sampling issues, or access restrictions.

Do not submit private, sensitive, confidential, or credential data. Do not include API keys, access tokens, passwords, or private account information in the submitted files. If the full dataset cannot be submitted because of size, privacy, copyright, platform terms, or access restrictions, submit a representative sample that shows the data structure and explain the limitation in the report or a README file.

## Network Analysis Requirements

All projects must include a meaningful network analysis component. A network visualisation alone is not sufficient. Your report should explain how the network was constructed and how the network analysis helps answer the project question or aim.

At minimum, the report should state:

- What the nodes represent.
- What the edges represent.
- Whether the network is directed or undirected, and why.
- Whether the network is weighted or unweighted, and why.
- Any filtering, thresholding, sampling, or preprocessing decisions used to construct the network.
- At least one relevant network measure, algorithm, or analytical method used to interpret the network.

The network analysis should be connected to the project aim. For example, if the project is about influential users, the report should explain why the chosen influence or centrality measure is appropriate. If the project is about communities, the report should explain how communities are detected and how they are interpreted.

For COSC 2671 projects, network structure and influence analysis must go beyond listing a centrality ranking. The report should interpret what the network structure or influence results reveal about user roles, communities, interaction patterns, information flow, or another relevant aspect of the project question.

## Perform the Analysis/Solve the Problem

The next component is to gather or obtain the necessary data, perform the analysis, and iterate on your approach. You are encouraged to use tools and techniques studied in this course, and you may also use relevant techniques from other courses, such as data visualisation, time series analysis, or machine learning, where these support the project aim.

The core data processing, network construction, NLP/text analysis, modelling, and assessed visualisation must be implemented primarily in Python.

You may use standard Python packages, such as pandas, NumPy, scikit-learn, NetworkX, NLTK, spaCy, matplotlib, seaborn, Plotly, or similar libraries. Package use should be appropriate and acknowledged where relevant. You must not copy an existing end-to-end solution.

Your report should explain why you selected each major analytical approach and how it helps answer the project question or aim. Exploratory work is encouraged, but the final report should focus on the methods and results that are most relevant to the project.
