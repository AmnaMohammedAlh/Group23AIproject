import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# List of domains
domains = [
    "Career", "Financial", "Spirituality",
    "Health & Fitness", "Personal Development",
    "Family",  "Social", "Fun & Recreation"
]
life_features = ["Career", "Financial", "Spiritual", "Physical", "Intellectual", "Family", "Social", "Fun"]

# Personality and Hobbies Options
personality_types = ["Extrovert", "Introvert", "Ambivert"]
hobby_options = ["Exercise", "Reading", "Art", "Writing", "Socializing"]

# Section 1: Generic Survey
st.title("Life Harmony Questionnare")

st.header("Step 1: Fill Out General Information")
age = st.number_input("Age", min_value=1, max_value=120, value=25, step=1)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
occupation = st.selectbox(
    "Occupation",
    ["Full-time", "Part-time", "Freelancer", "Student", "Unemployed"]
)
budget = st.number_input("Budget", min_value=0, value=1000, step=100)
allocated_time = st.slider("Time Allocated (hrs/week)", 1, 40, 10)
personality = st.selectbox("Personality Type", personality_types)

# Hobbies selection with option to add custom hobbies
hobby = st.selectbox("Which hobby resonates with you the most:", hobby_options)

# Store user inputs for reference
st.write("### Summary of Your Inputs:")
st.write(f"- Age: {age}")
st.write(f"- Gender: {gender}")
st.write(f"- Marital Status: {marital_status}")
st.write(f"- Occupation: {occupation}")
st.write(f"- Budget: {budget}")
st.write(f"- Time Allocated: {allocated_time} hours/week")
st.write(f"- Personality: {personality}")
st.write(f"- Hobbies: {hobby}")

# Initialize values for ratings
if "current_ratings" not in st.session_state:
    st.session_state.current_ratings = [0] * len(domains)  # Default value: 0
if "goal_ratings" not in st.session_state:
    st.session_state.goal_ratings = [0] * len(domains)  # Default value: 0

# Function to plot the radar chart
def plot_radar_chart(values, title):
    num_vars = len(domains)

    # Compute the angle for each domain
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    # Create a new list to avoid modifying the original `values`
    values_to_plot = values + [values[0]]  # Complete the loop for values

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values_to_plot, color="skyblue", alpha=0.4)
    ax.plot(angles, values_to_plot, color="blue", linewidth=2)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(domains, fontsize=12)
    ax.set_title(title, fontsize=16, pad=20)

    st.pyplot(fig)

# Section 2: Current Ratings
st.header("Step 2: Rate Your Current Satisfaction")
for i, domain in enumerate(domains):
    st.session_state.current_ratings[i] = st.slider(
        f"{domain}", 1, 10, st.session_state.current_ratings[i]
    )

# Display current radar chart
st.write("### Current Ratings Radar Chart")
plot_radar_chart(st.session_state.current_ratings, "Current Ratings")

# Section 3: Goal Ratings
st.header("Step 3: Set Your Goal Satisfaction Levels")
for i, domain in enumerate(domains):
    st.session_state.goal_ratings[i] = st.slider(
        f"Goal for {domain}", 1, 10, st.session_state.goal_ratings[i]
    )

# Display goal radar chart
st.write("### Goal Ratings Radar Chart")
plot_radar_chart(st.session_state.goal_ratings, "Goal Ratings")


if st.button("Submit and Get Recommendations"):
    # ----------------------------------------------------
    # Section 4: Compute Gaps and Assign Priorities
    st.header("Step 4: Analyze Gaps and Prioritize Domains")

    # Add a Submit button to calculate priorities and generate recommendations
    # Calculate gaps
    gaps = {
        domain: st.session_state.goal_ratings[i] - st.session_state.current_ratings[i]
        for i, domain in enumerate(domains)
    }

    # Normalize gaps (divide by max gap to get values between 0 and 1)
    max_gap = max(abs(val) for val in gaps.values())
    normalized_gaps = {key: abs(value) / max_gap for key, value in gaps.items()} if max_gap != 0 else {key: 0 for key in gaps}

    # Assign priorities
    if len(set(normalized_gaps.values())) == 1:  # All gaps are equal
        # Tie-breaking rule: Alphabetical order of domain names
        sorted_domains = sorted(normalized_gaps.keys())
        priorities = {
            key: "High" if key == sorted_domains[0] else "Medium" if key == sorted_domains[1] else "Low"
            for key in normalized_gaps
        }
    else:
        # Assign priorities based on thresholds
        thresholds = np.percentile(list(normalized_gaps.values()), [25.00, 66.67])
        priorities = {
            key: "Low" if normalized_gaps[key] <= thresholds[0] else
                 "Medium" if normalized_gaps[key] <= thresholds[1] else
                 "High"
            for key in normalized_gaps
        }

    # Display priorities
    st.write("### Domain Priorities")
    high_priority_domains = [key for key, value in priorities.items() if value == "High"]
    st.write(f"**High Priority Domains:** {high_priority_domains}")
    st.write(f"**Medium Priority Domains:** {[key for key, value in priorities.items() if value == 'Medium']}")
    st.write(f"**Low Priority Domains:** {[key for key, value in priorities.items() if value == 'Low']}")

    # ----------------------------------------------------
    # Section 5: Recommendations
    st.header("Step 5: Recommendations Based on Q-Table and k-NN")
    from sklearn.neighbors import NearestNeighbors

    marital_status_mapping = {"Single": 0, "Married": 1}
    occupation_mapping = {"Full-time": 0, "Part-time": 1, "Freelancer": 2, "Student": 3, "Unemployed": 4}
    personality_mapping = {"Extrovert": 0, "Introvert": 1, "Ambivert": 2}
    hobby_mapping = {"Exercise": 0, "Reading": 1, "Writing": 2, "Art": 3, "Socializing": 4}
    priority_mapping = {"Low": 0, "Medium": 1, "High": 2}

    # Load the Q-table and action mappings
    q_table = np.load("train_q_table.npy")  # Load the saved Q-table
    index_to_action = np.load("index_to_action.npy", allow_pickle=True).item()  # Load the index-to-action mapping
    encoded_training_states = np.load("encoded_training_states.npy")  # Load the saved encoded training states

    # Recommendation function using k-NN and Q-table
    def generate_recommendations_knn(
        user_state_vector, q_table, encoded_training_states, index_to_action, threshold=0.7, k=1
    ):
        """
        Generate recommendations for a user's encoded state using k-NN and the Q-table.

        Parameters:
        - user_state_vector: Encoded vector representing the user's input state.
        - q_table: Trained Q-table from the RL model.
        - encoded_training_states: Encoded states from the training dataset.
        - index_to_action: Mapping from action indices to descriptions.
        - threshold: Threshold for determining optimal actions.
        - k: Number of nearest neighbors to consider.

        Returns:
        - unique_recommendations: List of recommended actions.
        """
        # Reshape user state for compatibility
        user_state_vector = np.array(user_state_vector).reshape(1, -1)

        # Initialize and fit k-NN
        knn = NearestNeighbors(n_neighbors=k, metric="cosine")
        knn.fit(encoded_training_states)

        # Find nearest neighbors
        distances, indices = knn.kneighbors(user_state_vector)

        # Retrieve optimal actions from Q-table
        optimal_actions = []
        for idx in indices[0]:
            state_q_values = q_table[idx]
            max_q_value = np.max(state_q_values)
            actions = [
                index_to_action[action]
                for action in np.where(state_q_values >= threshold * max_q_value)[0]
            ]
            optimal_actions.extend(actions)

        # Get unique recommendations
        unique_recommendations = list(set(optimal_actions))
        return unique_recommendations

    # User's encoded state (obtained from prior steps)
    user_state_vector = [
        age,
        marital_status_mapping[marital_status],
        occupation_mapping[occupation],
        budget,
        personality_mapping[personality],
        hobby_mapping[hobby],
        *[priority_mapping[priorities[domain]] for domain in domains],  # Map priorities
    ]

    # Generate recommendations using k-NN and Q-table
    recommendations = generate_recommendations_knn(
        user_state_vector=user_state_vector,
        q_table=q_table,
        encoded_training_states=encoded_training_states,
        index_to_action=index_to_action,
        threshold=0.7,  # Define a suitable threshold
        k=4  # Number of nearest neighbors to consider
    )

    # Display recommendations
    st.write("### Your Recommendations")
    if recommendations:
        for action in recommendations:
            st.write(f"- {action}")
    else:
        st.write("No recommendations found matching your priorities.")


