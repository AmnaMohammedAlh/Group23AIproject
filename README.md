*Overview*
LifeHarmony is an AI-driven recommender system designed to guide users toward achieving a balanced and fulfilling life. It combines user inputs, logical modeling, and machine learning to identify gaps in life domains and provide actionable, personalized recommendations.
The application employs Reinforcement Learning (RL) and k-Nearest Neighbors (k-NN) to analyze user data, prioritize life areas, and recommend targeted actions. This repository includes the codebase, generated datasets, and model training artifacts required to replicate and enhance the system.

*Key Features*
- User Input:
Collects personal information (e.g., age, marital status, occupation).
- Gathers user ratings for current satisfaction and aspirational goals across eight life domains:
Career, Financial, Spirituality, Physical, Intellectual, Family, Social, Fun.
- Gap Analysis & Priority Mapping:
Identifies gaps between current and goal ratings.
Normalizes gaps and assigns priorities (High, Medium, Low) using percentile-based thresholds.

*Personalized Recommendations:*
Generates targeted recommendations using RL-based Q-tables and k-NN matching to nearest training states.
Actions are aligned with user attributes like personality, hobbies, and life priorities.

*Machine Learning Pipeline:*
- Data generation using probabilistic logic and behavioral modeling.
- Pre-trained Q-tables to associate encoded user states with high-impact recommendations.
- k-NN for similarity-based recommendations.

*Dataset Information*
Features
- Behavioral Features:
Age, gender, marital status, occupation, budget, time allocation, personality type, hobbies.
- Life Domain Ratings:
Current and goal satisfaction levels for eight domains.
- Priority Levels:
Derived priorities for each life domain based on normalized gaps.

Dataset Generation
Logical rules simulate real-world relationships (e.g., age vs. marital status, occupation vs. time allocation).

*Gap Analysis:*
Calculates gaps between current and goal ratings for each life domain.
Normalizes gaps to identify priority areas using statistical thresholds.
Assigns priorities: High, Medium, or Low.

*Recommendation System:*
Uses a pre-trained Q-table and k-NN for personalized suggestions.
Matches user states with training data to retrieve relevant actions.
Supports multi-label recommendations optimized for balance and relevance.

*Visualization:*
Radar charts to visualize current vs. goal satisfaction levels.
Displays prioritized domains and actionable suggestions.


*Codebase:*
app.py: Streamlit-based user interface.
model_training.py: RL and k-NN implementation for Q-table training and recommendations.
Generated Datasets:

1_generated_dataset.xlsx: Behavioral features generated using logical rules.
2_generated_dataset.xlsx: Extended dataset with current and goal ratings for life domains.
4_generated_dataset_with_recommendations.xlsx: Final dataset with tailored recommendations.
Model Artifacts:

train_q_table.npy: Trained Q-table used for RL-based recommendations.
index_to_action.npy: Mapping of action indices to recommendation descriptions.
encoded_training_states.npy: Encoded user states for k-NN matching.

*Documentation:*
Uploaded files provide in-depth insights into dataset generation, training processes, and logical mappings.
How It Works
Step 1: User Data Collection
Users provide inputs through the interactive Streamlit app.
Inputs include personal demographics, personality type, hobbies, budget, and time allocation.
Users rate their satisfaction and set goals for each life domain.
Step 2: Gap Analysis & Prioritization
Gaps are calculated between current satisfaction and goals.
A normalization process determines the relative importance of each domain.
Domains are categorized into High, Medium, or Low priority.
Step 3: Recommendation Generation
The user state is encoded as a vector using predefined mappings.
The encoded state is matched against a dataset of training states using k-NN.
The RL Q-table retrieves high-value actions based on the user's state.
Installation

*Future Improvements*
Dynamic Learning:
Incorporate real-time user feedback to update the Q-table dynamically.
Contextual Recommendations:
Use geographic, seasonal, or temporal data to make context-aware suggestions.
Collaborative Filtering:
Leverage user similarity for enhanced group-based recommendations.
