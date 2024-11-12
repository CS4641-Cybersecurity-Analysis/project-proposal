import helper.util as util
import numpy as np
import streamlit as st

# placeholder name until problem fully defined
st.title("CS 4641 Project Midterm Checkpoint - Cybersecurity Analysis")

st.header("Introduction & Background")
st.markdown(
    util.convert_in_text_citations(
        """
        The spread of the internet has given rise to malicious attacks meant to compromise computer systems. This has brought about the field of cybersecurity, concerned with defending electronic systems, networks, and data [1]. However, as cybersecurity systems become more advanced, so do the attacks they prevent, necessitating systems that learn to prevent new attacks themselves [1]. Currently, researchers use machine-learning techniques to identify new threats. Decision trees, convolutional neural networks, and support vector machines are most effective at detecting malware based on log data [2].

        Our dataset provides network traffic data labeled as benign or malicious. The features include source and destination IPs and ports; network protocol; associated service; duration; state of connection; whether it’s local; number of packets, bytes,and IP bytes sent; number of missed bytes; and whether the connection is malicious. The dataset was obtained from Kaggle: www.kaggle.com/datasets/agungpambudi/network-malware-detection-connection-analysis [3].
        """
    )
)

st.header("Problem Definition")
st.markdown(
    util.convert_in_text_citations(
        """
        Cybersecurity attacks, specifically network attacks, are becoming increasingly prevalent [4]. A lot of these attacks are through the network, where malware can be remotely delivered, accessed, updated, and executed. The first step in mitigating such attacks is identifying whether or not network activity is benign or malicious. We aim to identify such attacks, so further defense mechanisms (e.g. firewalls, challenges, cleaning infected devices) become more effective and efficient.
        """
    )
)

st.header("Methods")
st.markdown(
    util.convert_in_text_citations(
        """
        One preprocessing method, often used for network intrusion detection systems (NIDS), is normalizing non-categorical data to within a range from 0 to 1 [5]. This reduces outlier impact and feature weights dominating the learning process. The dataset is then standardized with a mean of 0 and a standard deviation of 1, which makes the data more consistent. These techniques resulted in significant improvements in accuracy for models like K-nearest neighbors and naive Bayes models. We implemented this using `sklearn.preprocessing.MinMaxScaler`.
        
        Another method is lightweight feature extraction, specifically, Correlation-based Feature Selection (CFS). The process selects a subset of features with the highest correlation with the observed variable and the least correlation with each other [6]. Network features are often redundant, so this helps with dimensionality reduction. 
        
        The third preprocessing method is label encoding, which converts categorical labels into numbers. Since our dataset has qualitative data, this approach ensures all features are considered. We implemented this using `sklearn.preprocessing.LabelEncoder`.

        We also used PCA, which we did not mention in our proposal, to transform our data and perform basic dimensionality reduction. By analyzing the components and the contributing features, we were able to drop a feature or 2 from our data. We use `sklearn.decomposition.PCA`.

        We propose three supervised ML algorithms: Support Vector Machine (SVM), Random Forest, and Convolutional Neural Networks (CNN). So far, we have implemented a model based on SVM.

        We implemented SVM as our first algorithm. Initially, we chose SVM due to its effectiveness with high-dimensional data [7], but this benefit did not come into play since we used PCA and our own knowledge of networking to reduce the number of dimensions. Instead of using the Radial Basis Function (RBF), we chose a linear kernel, aiming to establish a straightforward decision boundary. For the SVM implementation, we used the `sklearn.svm` library. There is a `C` regularization hyperparameter which defines how much we prefer margin between classes of the boundary over accuracy or performance. Large values of C correspond to choosing smaller-margin but better performing hyperplane boundary, while smaller values correspond to higher-margin but worse performing hyperplane boundary (in testing). We used k-fold cross validation to find a suitable value.

        Secondly, we expect that Random Forest will be a good algorithm to use because it allows us to identify the most important features to identify malware by providing feature importance scores [8]. The Random Forest algorithm performs well with imbalanced datasets, which is often the case with network logs. We will use the `sklearn.ensemble.RandomForestClassifier` library for implementation. 

        Finally, we expect that ANNs will be a good algorithm to use. We decided to use Artificial Neural Networks instead of our original selection of Convolutional Neural Networks due to the tabular nature of our data. The datasets identified do not have a spatial bias that image data tends to have, so an ANN’s simpler structure should properly handle the data’s features with increasing efficiency. Since ANNs do not have convolutional layers and instead have layers with fully connected neurons, they can process input features without needing to evaluate spatial relationships [9]. 
        """
    )
)

st.header("Results and Discussion")
st.markdown(
    """
    We utilized PCA for dimensionality reduction and to help visualize our dataset. We found that the data is highly learnable, with clear boundaries between classes for the most part. The dataset used was also relatively balanced, so we did not have to worry about relevant issues.
    """
)
st.image('./app/assets/img/pca3d.png')
st.image('./app/assets/img/num_labels.png')
st.markdown(
    util.convert_in_text_citations(
        """
        Our expected results for detecting malicious network traffic show strong performance across three models. SVM was anticipated to achieve 90-95% accuracy, with precision between 0.85-0.92 and recall from 0.88-0.95, excelling at identifying high-dimensional data ​[7][9]. We increased our expected accuracy from the previous 80-85% projected accuracy due to the learnable nature of the dataset. Previous users have been able to achieve near 100% accuracy without needing to finetune their models. Our SVM model achieved ~96% accuracy using Principal Component Analysis (PCA) to reduce the dimension of features to train on. 
        """
    )
)

st.image('./app/assets/img/conf_mat.png')
st.image('./app/assets/img/dec_bound.png')

st.markdown(
    util.convert_in_text_citations(
        """
        Accuracy: ~0.967

        Our SVM model has a fairly high overall accuracy, however it produces a lot of false positives, so predicting a network connection is malicious more often than it actually is. For this case, false positives is definitely a better problem to have than false negatives. For example, it would be fine to block benign connections if that means we also block actual malicious connections (especially since we can further reduce impact on benign connections by first sending challenges before blocking outright). The model did not end up being very inefficient, despite the large dataset, likely because of the pre-processing done to remove features and normalize data. We found that just one component was sufficient to score such an accuracy.

        We also had precision ~0.942 and recall ~0.9999, with a false positive rate of ~0.0712. This aligns with our findings of relatively high false positive rates, but overall very good performance.

        Random Forest is expected to reach 93-97% accuracy, with precision and recall both ranging from 0.92-0.96, thanks to its ability to handle imbalanced datasets and provide feature importance ​[5]. ANN should achieve 94-98% accuracy, with precision around 0.90-0.95 and recall from 0.88-0.94, excelling at pattern recognition in network traffic ​which makes it suitable for intrusion detection ​[10]. Each model has its unique strengths but faces challenges like scalability and interpretability; for example, while ANNs are powerful in detecting complex traffic patterns, their computational complexity might lead to slower inference times. Although Random Forest performs well with imbalanced data and provides feature importance scores, it can become computationally expensive when scaling up to larger datasets. Implementing methods like Parallel Random Forest or optimizing the number of decision trees can significantly reduce this overhead. Likewise, in terms of memory usage, SVM requires significant memory as it needs to store the entire dataset during the training process, which can lead to inefficiency, especially with large datasets. Although as previously mentioned, the preprocessing methods we used to reduce the dimension space likely aided in optimizing the model’s efficiency. 

        Our next steps for improving our detection system include incorporating more datasets which can capture a more diverse range of malicious activity. Improving the diversity in our data will result in more generalizable models that can more accurately recognize security compromises. Currently, our SVM model produces frequent false positives. By increasing the types of both malicious and normal behavior in network traffic, the model will be able to learn more generalized patterns instead of learning specific trends in one dataset. More data diversity will be especially beneficial to the next model we plan to implement, which is the Random Forest model. This system’s ensemble method relies on combining multiple decision trees which were trained on subsets of this data. By increasing the range of behavior shown, each tree can learn distinct trends and minimize the risk of overfitting to specific tendencies in one dataset.
        """
    )
)

st.header("Incorporating Proposal Feedback")
st.markdown(
    """
    To address our proposal feedback, we started by changing our citations from the actual Kaggle pages to the groups of researchers’ literature(s) that included the data since we were supposed to cite the literature and not the Kaggle page originally. We then addressed our target predictions of performance given the type of model we used when we realized that the algorithms we will be utilizing, like SVM and RF, far surpass the expectations we predicted via the applications of the same ML algorithms (like SVM and RF) on the initial dataset we used, so in turn we substantiated our predictions to reciprocate accuracies in the Kaggle notebooks utilizing said initial dataset. This leads into the next point of amending our utilization of CNN. We amended our proposal to focus our third and final algorithm on an ANN implementation that is not only more inherently utilized with tabular data unlike CNN’s, but also avoids the spatial bias in CNN’s where images may be treated as common structures of pattern to be learned. To address the lack of attention rendered to hyperparameter tuning in our proposal, we decided to test a few regularization hyperparameter values (0.2 → 1.0 through increments of 0.2) via k-fold cross validation, where we saw our accuracy scores only increase as the regularization hyperparameter C increased in value; therefore, we found C = 1.0 to be the best regularization value and used that to train our SVM model; this is not to discourage the other 4 hyperparameters, which all individually had almost as high of an accuracy as C = 1.0. Finally, with the Gantt chart, start dates and blockages have been addressed to where no blockages should occur.
    """
)

st.header("Gantt Chart")
st.image('./app/assets/img/gantt-chart-mid.png', "Project Gantt Chart")

st.header("Contribution Table")
st.dataframe(
    np.array([
        ['Shreya', 'Model Analysis'],
        ['Saim', 'SVM Model Implementation'],
        ['Vinay', 'Metrics Analysis, Data Visualization'],
        ['Timothy', 'Preprocessing Implementation, Data Visualization'],
        ['Olivia', 'Incorporate Proposal Feedback, Next Steps, Gantt Chart']
    ]),
    column_config={
        "0": 'Team Member',
        "1": 'Contributions',
    },
    width=1000
)

st.header("References")
references = []
references.append('[1] J. Martínez Torres, C. Iglesias Comesaña and P. J. García-Nieto, "Review: machine learning techniques applied to cybersecurity," International Journal of Machine Learning and Cybernetics, vol. 10, (10), pp. 2823-2836, 2019. Available: https://www.proquest.com/scholarly-journals/review-machine-learning-techniques-applied/docview/2920238591/se-2. [Accessed Oct. 03, 2024]')
references.append('[2] M.S. Akhtar and T. Feng, “Malware Analysis and Detection Using Machine Learning Algorithms,” Symmetry, vol. 14, (11), 2304, 2022. Available: https://www.mdpi.com/2073-8994/14/11/2304. [Accessed Oct. 03, 2024]')
references.append('[3] S. Garcia, A. Parmisano, & M. Jose Erquiaga. (2023). "Malware Detection in Network Traffic Data" [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/7285844')
references.append('[4] M. Abdullahi et al., “Detecting Cybersecurity Attacks in Internet of Things Using Artificial Intelligence Methods: A Systematic Literature Review,” Electronics, vol. 11, no. 2, p. 198, Jan. 2022, doi: https://doi.org/10.3390/electronics11020198.')
references.append('[5] Larriva-Novo, X., Villagrá, V. A., Vega-Barbas, M., Rivera, D., & Sanz Rodrigo, M. (2021). An IOT-focused intrusion detection system approach based on preprocessing characterization for cybersecurity datasets. Sensors, 21(2), 656. Available: https://doi.org/10.3390/s21020656')
references.append('[6] Soe, Y. N., Feng, Y., Santosa, P. I., Hartanto, R., & Sakurai, K. (2020, January 11). Towards a lightweight detection system for cyber attacks in the IOT environment using corresponding features. MDPI. https://www.mdpi.com/2079-9292/9/1/144')
references.append('[7] Karaman, Yunus, et al. “A Comparative Analysis of SVM, LSTM and CNN-RNN Models for the BBC News Classification.” SpringerLink, Springer International Publishing, 1 Jan. 1970. Available: https://link.springer.com/chapter/10.1007/978-3-031-26852-6_44. [Accessed Oct. 04, 2024]')
references.append('[8] Manish Choubisa, R. Doshi, N. Khatri, and Kamal Kant Hiran, “A Simple and Robust Approach of Random Forest for Intrusion Detection System in Cyber Security,” May 2022, doi: https://doi.org/10.1109/icibt52874.2022.9807766.')
references.append('[9] Vakili, Meysam, et al. “Performance Analysis and Comparison of Machine and Deep Learning Algorithms for IOT Data Classification.” arXiv.Org, 27 Jan. 2020. Available: https://arxiv.org/abs/2001.09636. [Accessed Oct. 04, 2024]')
references.append('[10] Makandar, A., & Patrot, A. (2015, December 1). Malware analysis and classification using Artificial Neural Network. IEEE Xplore. https://ieeexplore.ieee.org/Xplore/home.jsp')

for reference in references:
    reference_html = util.create_reference(reference)
    st.html(reference_html)
