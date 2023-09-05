from copy import deepcopy

import streamlit as st

import matplotlib.pyplot as plt
import numpy as np
import streamlit_functions
import seaborn as sns
import streamlit_conf_all_data as streamlit_conf

# st.set_page_config(layout='wide')

tab1, tab2 = st.tabs(["Field Descriptions", "Results"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        # Description text
        description_text = """
        ## Fields Description
        - **Select Dataset**: Authors that appeared in the TRAIN set (other SHUTs but same authors) or did not appear in the TRAIN set (TEST set Authors).
        - **Select Authors**: Which authors are included in the evaluation.
        - **Predict the Highest Score**: Predict the highest score in each row or use a thresholdPredict the highest score in each row or use a threshold.
        - **Select Threshold**: Threshold score above which "same author" is predicted.
        - **Number of SHUTS per Author**: The number of times the predictions will be calculated (for different random samples) and the averages matrix displayed. For example, if 20 SHUTS are selected, 20 matrices will be calculated, and the average score for each cell (i, j) will be displayed.
        """

        # Display description text
        st.markdown(description_text)
    with col2:
        dataset_type = 'TEST'


        # def load_train_authors():
        #     return streamlit_conf.get_data('TRAIN')

        def load_test_authors(test_category):
            return streamlit_conf.get_data('TEST', test_category)


        categories = ['Drashot', 'Hasidut', 'Maarachot', 'Mitzvots', 'Halacha', 'Bavli', 'Mishna', 'Tanach']
        test_category = st.selectbox('Select Test Genre', categories, index=1)

        # # Load train_authors and test_authors on the first run of the script
        # if 'train_authors' not in st.session_state:
        #     st.session_state.train_authors = load_train_authors()

        if 'test_authors' not in st.session_state:
            st.session_state.test_authors = load_test_authors(test_category)

        # train_authors = st.session_state.train_authors
        test_authors = st.session_state.test_authors

        # Select authors
        selected_authors_test = st.multiselect('Select Test Authors', test_authors)

        #
        # selected_model = st.selectbox('Select Model', model_types)

        # Checkbox for choosing the highest score near the threshold
        show_highest_score = st.checkbox("Predict the highest Score")

        if not show_highest_score:
            # Select threshold
            threshold = st.slider('Select Threshold', 0.0, 1.0, 0.7)

        # Select threshold (if show_highest_score is not marked)
        if show_highest_score:
            threshold = None

        # Select threshold
        numer_of_matrices = st.slider('Number of SHUTS per Author', 1, 50, 5)
        # Execute code based on selections
        run_btn = st.button('Run Code')
        if run_btn:
            st.markdown("<span style='color:blue; font-weight:bold;'>View Results Tab</span>", unsafe_allow_html=True)

            with tab2:
                model_types = ['Shuts', 'Rambam', 'MusarRambamSa', 'ShutsMusarRambamSa']
                col1, col2 = st.columns(2)
                for selected_model in model_types:
                    # Select the appropriate column for plotting
                    if selected_model in ['Shuts', 'MusarRambamSa']:
                        col = col1
                    else:
                        col = col2
                    similarity_matrices = streamlit_conf.read_scores(dataset_type, selected_model, test_category)

                    authors = deepcopy(test_authors)
                    selected_authors = deepcopy(selected_authors_test)

                    val_authors_dict = {value: index for index, value in enumerate(authors)}
                    values_list = [val_authors_dict[author] for author in selected_authors]
                    val_authors_reserved = [author[::-1] for author in selected_authors]

                    cm = streamlit_functions.generate_cm(similarity_matrices, numer_of_matrices)
                    cm = np.round(cm, decimals=2)

                    cm = cm[np.ix_(values_list, values_list)]
                    fig, ax = plt.subplots(figsize=(5, 5))

                    heat_map_figure = sns.heatmap(cm, annot=True, cmap="BuPu", fmt='g',
                                                  xticklabels=val_authors_reserved,
                                                  yticklabels=val_authors_reserved, cbar=False, linewidths=2,
                                                  annot_kws={"fontsize": 12})
                    if selected_model == 'ShutsMusarRambamSa':
                        title = 'שות + מוסר +רמבם + שולחן ערוך'
                    elif selected_model == 'MusarRambamSa':
                        title = 'מוסר +רמבם + שולחן ערוך'
                    elif selected_model == 'Shuts':
                        title = 'שות'
                    if selected_model == 'Rambam':
                        title = 'רמבם'
                    ax.set_title(title[::-1], fontsize=12)

                    heat_map_figure.set_xticklabels(heat_map_figure.get_xticklabels(), rotation=90)
                    heat_map_figure.set_yticklabels(heat_map_figure.get_yticklabels(), rotation=0)

                    if show_highest_score:
                        # Mark the highest value in each row with a red square
                        for i in range(len(cm)):
                            max_value = np.max(cm[i])
                            max_index = np.argmax(cm[i])
                            ax.add_patch(plt.Rectangle((max_index, i), 1, 1, fill=False, edgecolor='red', lw=4))
                    else:
                        # Mark values greater than threshold
                        indices = np.where(cm > threshold)
                        for i, j in zip(indices[0], indices[1]):
                            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=4))

                    # Increase font size for x and y axis labels
                    ax.tick_params(axis='x', labelsize=12)
                    ax.tick_params(axis='y', labelsize=12)
                    with col:
                        st.pyplot(fig)
