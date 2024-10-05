import helper.util as util
import numpy as np
import streamlit as st

# placeholder name until problem fully defined
st.title("CS 4641 Project Proposal - Cybersecurity Analysis")

st.header("Introduction & Background")
st.markdown(
    util.convert_in_text_citations(
        """
        The spread of the internet has given rise to malicious attacks meant to compromise computer systems. Good cybersecurity systems, concerned with defending electronic systems, networks, and data, require learning constantly newer, advanced attacks [1]. Machine-learning algorithms, such as decision trees, convolutional neural networks, and support vector machines, are most effective at detecting malware based on log data [2].

        Our dataset provides network traffic data with labels of several malicious attacks and benign. The features include source and destination IPs and ports; network protocol; associated service; duration; state of connection; locality; number of packets, bytes, and IP bytes sent; number of missed bytes; and the label. The dataset was obtained from Kaggle [3].
        """
    )
)

st.header("Problem Definition")
st.markdown(
    util.convert_in_text_citations(
        """
        Cybersecurity attacks, specifically network attacks, are becoming exponentially prevalent [4]. Plenty of these attacks are through the network, where malware can be remotely delivered, accessed, updated, and executed. These attacks demand security, and the first step is classifying network activity. We aim to identify such attacks, so further defense mechanisms (e.g. firewalls, challenges, cleaning infected devices) become more efficient.
        """
    )
)

st.header("Methods")
st.markdown(
    util.convert_in_text_citations(
        """
        One potential preprocessing method, often used for network intrusion detection systems (NIDS), is normalizing non-categorical data to within a range from 0 to 1 [5]. This reduces outlier impact and feature weights dominating the learning process. The dataset is then standardized with a mean of 0 and a standard deviation of 1, which makes the data more consistent. These techniques resulted in significant improvements in accuracy for models like K-nearest neighbors and naive Bayes models. 
        
        Another method is lightweight feature extraction, specifically, Correlation-based Feature Selection (CFS). The process selects a subset of features with the highest correlation with the observed variable and the least correlation with each other [6]. Network features are often redundant, so this helps with dimensionality reduction. 
        
        The third preprocessing method is label encoding, which converts categorical labels into numbers. Since our dataset has qualitative data, this approach ensures all features are considered.

        We propose three supervised ML algorithms: Support Vector Machine (SVM), Random Forest, and Convolutional Neural Networks (CNN). 

        SVM should be an effective algorithm because it works well with high-dimensional data [7]. Additionally, SVM can use the Radial Basis Function (RBF) to transform data into a higher dimensional space for non-linear boundaries. For example, the duration and number of bytes exchanged in a network log might indicate unusual activity only when combined with a specific protocol. We can use the `sklearn.svm` library for implementation. 

        Secondly, we expect that Random Forest will be a good algorithm to use because it allows us to identify the most important features to identify malware by providing feature importance scores [8]. The Random Forest algorithm performs well with imbalanced datasets, which is often the case with network logs. We will use the `sklearn.ensemble.RandomForestClassifier` library for implementation. 

        Finally, we expect that CNNs will be a good algorithm to use, as they are great at identifying patterns within data and handling noisy features [9]. We plan to use the Keras library to implement this algorithm.
        """
    )
)

st.header("Results and Discussion")
st.markdown(
    util.convert_in_text_citations(
        """
        Our expected results for detecting malicious network traffic show strong performance across three models. We use accuracy, precision, and recall as our main performance metrics. SVM is anticipated to achieve 80-90% on all 3, excelling at identifying high-dimensional data [7][9]. Random Forest is expected to reach mid 80s accuracy and mid 90s precision and recall, thanks to its ability to handle imbalanced datasets and provide feature importance [5]. CNN should achieve 84-88% accuracy, with precision at 90-95% and recall from 88-94%, excelling at pattern recognition in network traffic which makes it suitable for intrusion detection [7][9].
        """
    )
)

st.header("Gantt Chart")
st.image('./app/assets/img/gantt-chart.png', "Project Gantt Chart")

st.header("Contribution Table")
st.dataframe(
    np.array([
        ['Shreya', 'Introduction/Background, Presentation'],
        ['Saim', 'Methods'],
        ['Vinay', 'Results/Discussion'],
        ['Timothy', 'Problem Definition, Streamlit, Review'],
        ['Olivia', 'Processing methods, Gantt Chart']
    ]),
    column_config={
        "0": 'Team Member',
        "1": 'Contributions',
    }
)

st.header("References")
references = []
references.append('[1] J. Martínez Torres, C. Iglesias Comesaña and P. J. García-Nieto, "Review: machine learning techniques applied to cybersecurity," International Journal of Machine Learning and Cybernetics, vol. 10, (10), pp. 2823-2836, 2019. Available: https://www.proquest.com/scholarly-journals/review-machine-learning-techniques-applied/docview/2920238591/se-2. [Accessed Oct. 03, 2024]')
references.append('[2] M.S. Akhtar and T. Feng, “Malware Analysis and Detection Using Machine Learning Algorithms,” Symmetry, vol. 14, (11), 2304, 2022. Available: https://www.mdpi.com/2073-8994/14/11/2304. [Accessed Oct. 03, 2024]')
references.append('[3] A. Pambudi, "Malware Detection in Network Traffic Data", Kaggle, 2020. Available: https://www.kaggle.com/datasets/agungpambudi/network-malware-detection-connection-analysis')
references.append('[4] M. Abdullahi et al., “Detecting Cybersecurity Attacks in Internet of Things Using Artificial Intelligence Methods: A Systematic Literature Review,” Electronics, vol. 11, no. 2, p. 198, Jan. 2022, doi: https://doi.org/10.3390/electronics11020198.')
references.append('[5] Larriva-Novo, X., Villagrá, V. A., Vega-Barbas, M., Rivera, D., & Sanz Rodrigo, M. (2021). An IOT-focused intrusion detection system approach based on preprocessing characterization for cybersecurity datasets. Sensors, 21(2), 656. Available: https://doi.org/10.3390/s21020656')
references.append('[6] Soe, Y. N., Feng, Y., Santosa, P. I., Hartanto, R., & Sakurai, K. (2020, January 11). Towards a lightweight detection system for cyber attacks in the IOT environment using corresponding features. MDPI. https://www.mdpi.com/2079-9292/9/1/144')
references.append('[7] Karaman, Yunus, et al. “A Comparative Analysis of SVM, LSTM and CNN-RNN Models for the BBC News Classification.” SpringerLink, Springer International Publishing, 1 Jan. 1970. Available: https://link.springer.com/chapter/10.1007/978-3-031-26852-6_44. [Accessed Oct. 04, 2024]')
references.append('[8] Manish Choubisa, R. Doshi, N. Khatri, and Kamal Kant Hiran, “A Simple and Robust Approach of Random Forest for Intrusion Detection System in Cyber Security,” May 2022, doi: https://doi.org/10.1109/icibt52874.2022.9807766.')
references.append('[9] Vakili, Meysam, et al. “Performance Analysis and Comparison of Machine and Deep Learning Algorithms for IOT Data Classification.” arXiv.Org, 27 Jan. 2020. Available: https://arxiv.org/abs/2001.09636. [Accessed Oct. 04, 2024]')

for reference in references:
    reference_html = util.create_reference(reference)
    st.html(reference_html)
